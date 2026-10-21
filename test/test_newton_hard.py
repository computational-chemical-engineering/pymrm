"""Hard problems for :func:`pymrm.newton`: what each option is for.

Every test is a small, deterministic model on which plain Newton fails or
gives a wrong verdict, and the option that fixes it. The numbers (statuses,
iteration counts) were measured when the options were introduced; they pin
the behaviour the documentation describes.
"""

import warnings

import numpy as np
import pytest
from scipy.sparse import csc_array, diags_array
from scipy.sparse.linalg import splu

from pymrm import NumJac, construct_div, construct_grad, newton


def _slab_operators(n, nu=0):
    """``div(-grad c)`` with symmetry at 0 and ``c = 1`` at 1."""
    r_f = np.linspace(0.0, 1.0, n + 1)
    shape = (n, 1)
    bc = ({"a": 1.0, "b": 0.0, "d": 0.0}, {"a": 0.0, "b": 1.0, "d": 1.0})
    grad, grad_bc = construct_grad(shape, r_f, bc=bc, axis=0)
    div = construct_div(shape, r_f, nu=nu, axis=0)
    return shape, r_f, (div @ -grad).tocsc(), (div @ -grad_bc).toarray().reshape(-1, 1)


def _pellet(rate, n=100, nu=0):
    shape, r_f, jac_const, g_const = _slab_operators(n, nu)
    numjac = NumJac(shape)

    def fun(c):
        g_r, j_r = numjac(rate, c.reshape(shape))
        return g_const + jac_const @ c.reshape(-1, 1) + g_r.reshape(-1, 1), jac_const + j_r

    return fun, shape, 0.5 * (r_f[:-1] + r_f[1:])


def _weisz_hicks(phi, gamma, beta, n=60):
    """Non-isothermal sphere, Prater temperature ``1 + beta (1 - c)``."""
    def rate(c):
        t = beta * (1.0 - c)
        return phi**2 * c * np.exp(gamma * t / (1.0 + t))
    return _pellet(rate, n=n, nu=2)[:2]


# ---------------------------------------------------------------------------
# Convergence verdict: trace unknowns and the backward-error gate
# ---------------------------------------------------------------------------

def _two_fields(x):
    """Field 0 solves x^2 = 1, field 1 (trace) solves x^2 = 1e-20, per cell."""
    target = np.array([1.0, 1e-20])
    g = (x**2 - target).ravel()
    return g, diags_array((2.0 * x).ravel()).tocsc()


def test_backward_tol_fixes_the_absolute_tolerance_trap_per_field():
    x0 = np.tile([1.0 + 1e-9, 1e-8], (3, 1))
    plain = newton(_two_fields, x0)
    assert plain.success and plain.x[0, 1] / 1e-10 > 40.0   # trace field 50x wrong
    assert plain.backward_error > 1e-4                      # ... and the result says so
    gated = newton(_two_fields, x0, backward_tol=1e-8)
    assert gated.success and gated.status == 0
    # without scale a field is judged at no less than 1e-6 of the largest unknown
    np.testing.assert_allclose(gated.x[:, 1], 1e-10, rtol=1e-3)
    np.testing.assert_allclose(gated.x[:, 0], 1.0, rtol=1e-12)
    scaled = newton(_two_fields, x0, backward_tol=1e-8, scale=np.array([1.0, 1e-10]))
    np.testing.assert_allclose(scaled.x[:, 1], 1e-10, rtol=1e-8)


def test_backward_error_judges_a_dead_zone_against_its_field():
    """A reaction dead zone (c ~ 1e-30 next to c = 1) is no reason to iterate on."""
    fun, shape, _ = _pellet(lambda c: 900.0 * c, n=100)       # phi = 30, first order
    res = newton(fun, np.ones(shape), backward_tol=1e-8)
    assert res.status == 0 and res.nit <= 3


# ---------------------------------------------------------------------------
# Overshoot below zero: clipped versus regularised kinetics
# ---------------------------------------------------------------------------

PHI = 30.0
EPS = 1e-6


def _dead_zone_exact(z):
    """Slab, rate phi^2 c^(1/2): c = (phi^2/12)^2 (z - z0)^4 beyond z0 = 1 - sqrt(12)/phi."""
    z0 = 1.0 - np.sqrt(12.0) / PHI
    return np.where(z > z0, (PHI**2 / 12.0) ** 2 * np.clip(z - z0, 0.0, None) ** 4, 0.0)


def _clipped(c):
    return PHI**2 * np.sqrt(np.maximum(c, 0.0))


def _regularised(c, eps=EPS):
    return PHI**2 * c * (c**2 + eps**2) ** -0.25


def test_clipped_half_order_kinetics_defeat_plain_newton():
    fun, shape, _ = _pellet(_clipped)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = newton(fun, np.ones(shape))
    assert not res.success        # full steps cycle around the dead-zone edge


def test_regularised_kinetics_with_monotone_damping_converge():
    fun, shape, z = _pellet(_regularised)
    res = newton(fun, np.ones(shape), damping="monotone", backward_tol=1e-8)
    assert res.status == 0 and res.nit <= 20
    assert np.max(np.abs(res.x.ravel() - _dead_zone_exact(z))) < 2e-3
    # eps changes the model only where c ~ eps: tenfold smaller leaves it unchanged
    fun_small, _, _ = _pellet(lambda c: _regularised(c, EPS / 10))
    res_small = newton(fun_small, np.ones(shape), damping="monotone", backward_tol=1e-8)
    assert res_small.status == 0
    assert np.max(np.abs(res_small.x - res.x)) < 1e-6


def test_a_bound_at_zero_stalls_where_the_regularised_rate_converges():
    fun, shape, _ = _pellet(_clipped)
    bounded = newton(fun, np.zeros(shape), damping="monotone", bounds=0.0,
                     backward_tol=1e-8, stagnation=True)
    assert not bounded.success and bounded.status == 8   # held by the bound
    fun_reg, _, _ = _pellet(_regularised)
    free = newton(fun_reg, np.zeros(shape), damping="monotone", backward_tol=1e-8)
    assert free.status == 0


# ---------------------------------------------------------------------------
# Arrhenius pellets: ignition, non-finite steps and multiple steady states
# ---------------------------------------------------------------------------

def test_guarded_rescues_a_full_step_that_overflows():
    fun, shape = _weisz_hicks(phi=1.0, gamma=30.0, beta=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = newton(fun, np.ones(shape), maxfev=100)
        guarded = newton(fun, np.ones(shape), maxfev=100, damping="guarded")
    assert plain.status == 4                           # non-finite residual
    assert guarded.status == 0 and guarded.x.min() < 1e-3   # ignited branch


def test_guarded_takes_the_same_steps_as_plain_newton_when_these_work():
    fun, shape = _weisz_hicks(phi=1.0, gamma=20.0, beta=0.3)
    plain = newton(fun, np.ones(shape))
    guarded = newton(fun, np.ones(shape), damping="guarded")
    assert plain.status == guarded.status == 0 and plain.nit == guarded.nit
    np.testing.assert_allclose(guarded.x, plain.x, rtol=1e-12, atol=1e-14)


def test_multiple_steady_states_follow_the_initial_guess():
    """Weisz-Hicks sphere in its multiplicity window: damping does not choose."""
    fun, shape = _weisz_hicks(phi=0.4, gamma=20.0, beta=0.6)
    for damping in ("none", "guarded"):
        cold = newton(fun, np.ones(shape), damping=damping)
        hot = newton(fun, np.zeros(shape), damping=damping)
        assert cold.success and hot.success
        assert cold.x.min() > 0.9 and hot.x.min() < 1e-3


def test_monotone_damping_is_invariant_to_row_and_column_scaling():
    fun, shape = _weisz_hicks(phi=0.5, gamma=30.0, beta=0.5)
    base = newton(fun, np.ones(shape), damping="monotone", maxfev=60)

    def rows(c):  # the same equations in other units
        g, jac = fun(c)
        return 1e6 * g, 1e6 * jac

    scaled_rows = newton(rows, np.ones(shape), damping="monotone", maxfev=60)
    assert [h["damping"] for h in base.history] == pytest.approx(
        [h["damping"] for h in scaled_rows.history], rel=1e-8)
    np.testing.assert_allclose(scaled_rows.x, base.x, atol=1e-10)



def test_scale_equals_an_explicit_change_of_variables():
    """Two fields of very different size: scale=S is the same as solving for x/S."""
    fun1, shape1 = _weisz_hicks(phi=0.5, gamma=30.0, beta=0.5, n=30)
    big = 1e4

    def fun(x):                       # field 0: the pellet c; field 1: T = big * (1 + c)
        g1, j1 = fun1(x[:, :1])
        g2 = x[:, 1] - big * (1.0 + x[:, 0])
        jac = np.block([[j1.toarray(), np.zeros((30, 30))],
                        [-big * np.eye(30), np.eye(30)]])
        perm = np.arange(60).reshape(2, 30).T.ravel()      # field-major -> cell-major
        return np.concatenate([np.ravel(g1), g2])[perm], csc_array(jac[np.ix_(perm, perm)])

    s = np.array([1.0, big])
    x0 = np.column_stack([np.ones(30), 2 * big * np.ones(30)])
    scaled = newton(fun, x0, scale=s, damping="monotone", maxfev=60)

    def in_y(y):                      # the same problem in y = x / s
        g, jac = fun(y * s)
        return g, jac @ diags_array(np.tile(s, 30))
    explicit = newton(in_y, x0 / s, damping="monotone", maxfev=60)
    assert scaled.status == explicit.status == 0 and scaled.nit == explicit.nit
    assert [h["damping"] for h in scaled.history] == pytest.approx(
        [h["damping"] for h in explicit.history], rel=1e-6)
    np.testing.assert_allclose(scaled.x, explicit.x * s, rtol=1e-8, atol=1e-12)


# ---------------------------------------------------------------------------
# Failure is reported, not raised
# ---------------------------------------------------------------------------

def test_singular_jacobian_is_a_status():
    shape, _, jac_const, _ = _slab_operators(20)
    neumann = jac_const.copy()
    neumann[-1, -1] -= neumann[-1, -1] + neumann[-1, -2]   # make the last row zero-sum

    def fun(c):
        return neumann @ c.ravel() - 1.0, neumann
    res = newton(fun, np.zeros(shape))
    assert not res.success and res.status == 5


def test_double_root_is_not_reported_converged_with_backward_tol():
    def f(x):
        return np.array([x[0] ** 2]), csc_array([[2.0 * x[0]]])
    assert newton(f, np.array([1.0])).success            # plain step test
    gated = newton(f, np.array([1.0]), backward_tol=1e-8, maxfev=60)
    assert not gated.success and gated.status == 7


def test_stagnation_floor_ends_the_iteration():
    fun, shape, _ = _pellet(lambda c: 100.0 * c**2)
    without = newton(fun, np.ones(shape), tol=1e-20, maxfev=40)   # below round-off
    assert without.status == 2 and without.nit == 40
    floor = newton(fun, np.ones(shape), tol=1e-20, maxfev=40, stagnation=True,
                   backward_tol=1e-8)
    assert floor.status == 1 and floor.success and floor.nit <= 25


def test_recoverable_exception_at_a_trial_point_shortens_the_step():
    def f(x):
        with np.errstate(invalid="raise", divide="raise"):
            return np.array([np.log(x[0]) + 5.0]), csc_array([[1.0 / x[0]]])
    with pytest.raises(FloatingPointError):
        newton(f, np.array([1.0]))                         # full step to x = -4
    res = newton(f, np.array([1.0]), damping="guarded")
    assert res.status == 0 and res.x[0] == pytest.approx(np.exp(-5.0))
    assert res.history[0]["damping"] < 1.0


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

def test_bounds_broadcast_per_field_and_keep_the_iterates_inside():
    seen = []

    def f(x):
        seen.append(x.copy())
        # field 0: x^2 = 4 (root 2), field 1: x = -3 (unbounded)
        g = np.stack([x[:, 0] ** 2 - 4.0, x[:, 1] + 3.0], axis=1).ravel()
        jac = diags_array(np.stack([2.0 * x[:, 0], np.ones(len(x))], axis=1).ravel())
        return g, jac.tocsc()
    x0 = np.tile([0.1, 0.0], (4, 1))
    res = newton(f, x0, bounds=(np.array([0.0, -np.inf]), None))
    assert res.success
    np.testing.assert_allclose(res.x, np.tile([2.0, -3.0], (4, 1)))
    assert min(s[:, 0].min() for s in seen) > 0.0


def test_initial_guess_outside_the_bounds_is_projected_with_a_warning():
    def f(x):
        return x**2 - 4.0, diags_array(2.0 * x).tocsc()
    with pytest.warns(RuntimeWarning, match="outside the bounds"):
        res = newton(f, np.array([-1.0, 3.0]), bounds=(0.5, 2.5))
    assert res.success and np.allclose(res.x, 2.0)


def test_a_bound_that_cuts_off_the_solution_is_reported():
    def f(x):
        return x + 1.0, csc_array(np.eye(1))
    res = newton(f, np.array([1.0]), bounds=0.0, maxfev=30)
    assert not res.success and res.status == 8 and res.x[0] > 0.0


# ---------------------------------------------------------------------------
# Lagged quantities and user factorisations
# ---------------------------------------------------------------------------

def test_refresh_updates_a_lagged_coefficient():
    state = {"v": 2.0}

    def f(x):  # x^2 = v, with v = 2 + 0.1 x lagged outside the Jacobian
        return np.array([x[0] ** 2 - state["v"]]), csc_array([[2.0 * x[0]]])

    def refresh(x):
        state["v"] = 2.0 + 0.1 * x[0]
    res = newton(f, np.array([1.0]), refresh=refresh, tol=1e-12, maxfev=50)
    assert res.success
    assert res.x[0] == pytest.approx((0.1 + np.sqrt(0.01 + 8.0)) / 2.0, rel=1e-10)


class _SpluFactorize:
    def __init__(self):
        self.calls = 0

    def factorize(self, jac):
        self.calls += 1
        return splu(csc_array(jac))


def test_factorize_protocol_takes_part_in_the_damping_test():
    fun, shape = _weisz_hicks(phi=0.5, gamma=30.0, beta=0.5)
    solver = _SpluFactorize()
    res = newton(fun, np.ones(shape), solver=solver, damping="monotone", maxfev=60)
    ref = newton(fun, np.ones(shape), damping="monotone", maxfev=60, row_scaling=False)
    assert res.status == ref.status and res.nit == ref.nit
    assert solver.calls == res.nfact
    np.testing.assert_allclose(res.x, ref.x, atol=1e-10)
