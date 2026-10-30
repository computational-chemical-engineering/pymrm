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

from pymrm import NewtonBreakdown, NumJac, construct_div, construct_grad, newton


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
    plain = newton(_two_fields, x0, damping="none", backward_tol=None, stagnation=False)
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


def test_regularised_rate_beats_a_bound_at_zero_and_clipping():
    fun, shape, _ = _pellet(_clipped)
    fun_reg, _, _ = _pellet(_regularised)
    clipped = newton(fun, np.ones(shape), damping="monotone")
    bounded = newton(fun, np.ones(shape), damping="monotone", bounds=0.0)
    free = newton(fun_reg, np.ones(shape), damping="monotone")
    assert free.status == 0 and free.nit <= 16
    assert bounded.nit > 1.5 * free.nit and clipped.nit > 3 * free.nit


# ---------------------------------------------------------------------------
# Arrhenius pellets: ignition, non-finite steps and multiple steady states
# ---------------------------------------------------------------------------

def test_guarded_rescues_a_full_step_that_overflows():
    fun, shape = _weisz_hicks(phi=1.0, gamma=30.0, beta=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NewtonBreakdown) as plain:     # non-finite residual
            newton(fun, np.ones(shape), maxfev=100, damping="none")
        guarded = newton(fun, np.ones(shape), maxfev=100, damping="guarded")
    assert plain.value.result.status == 4
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

def test_singular_jacobian_raises_with_the_result_attached():
    jac = csc_array(np.array([[1.0, -1.0], [-1.0, 1.0]]))     # exactly singular

    def fun(x):
        return jac @ x - np.array([1.0, 0.0]), jac
    for row_scaling in (False, True):
        with pytest.raises(NewtonBreakdown) as err:
            newton(fun, np.array([0.5, 0.25]), row_scaling=row_scaling)
        assert err.value.result.status == 5
        np.testing.assert_array_equal(err.value.result.x, [0.5, 0.25])
    with pytest.raises(ValueError):          # as up to 2.5.0
        newton(fun, np.zeros(2))
    with pytest.raises(np.linalg.LinAlgError):
        newton(lambda x: (jac.toarray() @ x - 1.0, jac.toarray()), np.zeros(2))


def test_nearly_singular_jacobian_is_not_a_success():
    shape, _, jac_const, _ = _slab_operators(20)
    neumann = jac_const.copy()
    neumann[-1, -1] -= neumann[-1, -1] + neumann[-1, -2]   # zero-sum rows, up to round-off

    def fun(c):
        return neumann @ c.ravel() - 1.0, neumann
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = newton(fun, np.zeros(shape))
    assert not res.success


def test_double_root_converges_to_the_scale_of_the_problem():
    """x^2 = 0 from x = 1: linear convergence, judged against the initial scale."""
    def f(x):
        return np.array([x[0] ** 2]), csc_array([[2.0 * x[0]]])
    plain = newton(f, np.array([1.0]), damping="none", backward_tol=None)
    assert plain.success and plain.x[0] > 1e-9          # stops on the step, x ~ 1e-8
    gated = newton(f, np.array([1.0]), maxfev=100)
    assert gated.success and abs(gated.x[0]) < 1e-12


def test_stagnation_floor_ends_the_iteration():
    fun, shape, _ = _pellet(lambda c: 100.0 * c**2)
    # the round-off floor of the step is about 1e-15, within 100 tol = 1e-14
    without = newton(fun, np.ones(shape), tol=1e-16, maxfev=40, stagnation=False)
    assert without.status == 2 and without.nit == 40
    floor = newton(fun, np.ones(shape), tol=1e-16, maxfev=40, stagnation=True,
                   backward_tol=1e-8)
    assert floor.status == 1 and floor.success and floor.nit <= 25


@pytest.mark.parametrize("damping", ["monotone", "armijo"])
def test_damped_steps_at_a_round_off_floor_count_for_stagnation(damping):
    """Noise of 5e-11 above tol = 1e-12: the stagnation rule ends the iteration
    within a few steps although the damping test fails on the noise (measured on
    a fuel-cell model: 93 iterations instead of 8 when damped steps reset it)."""
    rng = np.random.default_rng(1)
    a = diags_array(np.full(200, 2.0)).tocsc()

    def noisy(u):
        return a @ (u - 1.0) + 5e-11 * rng.standard_normal(200), a

    res = newton(noisy, np.full(200, 5.0), tol=1e-12, damping=damping, stagnation=True,
                 backward_tol=None, maxfev=100)
    assert res.status == 1 and res.nit <= 30     # 100 (maxfev) when damped steps reset it


def test_recoverable_exception_at_a_trial_point_shortens_the_step():
    def f(x):
        with np.errstate(invalid="raise", divide="raise"):
            return np.array([np.log(x[0]) + 5.0]), csc_array([[1.0 / x[0]]])
    with pytest.raises(FloatingPointError):
        newton(f, np.array([1.0]), damping="none")         # full step to x = -4
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


def test_a_solution_near_a_bound_still_converges_in_one_step():
    """Linear sphere pellet, phi = 300: the solution is below 1e-100 at the centre.

    Only a step that crosses the bound is shortened, so the first full step,
    which lands inside (up to round-off), is taken as plain Newton takes it.
    """
    phi2 = 300.0**2
    fun0, shape, _ = _pellet(lambda c: phi2 * c, n=200, nu=2)
    iterates = []

    def fun(c):
        iterates.append(np.array(c, dtype=float).ravel())
        return fun0(c)

    plain_one = newton(fun0, np.ones(shape), maxfev=1).x
    one = newton(fun, np.ones(shape), bounds=(0.0, np.inf), maxfev=1)
    # the same step; its round-off crossings (7e-11 of a distance of 1) are
    # mirrored into the bound
    assert np.max(np.abs(plain_one)[plain_one < 0.0]) < 1e-8
    np.testing.assert_allclose(one.x, np.abs(plain_one), rtol=0.0, atol=1e-15)
    plain = newton(fun0, np.ones(shape), tol=1e-13)
    iterates.clear()
    res = newton(fun, np.ones(shape), bounds=(0.0, np.inf), tol=1e-13)
    assert res.status == 0 and res.nit == plain.nit
    assert min(float(np.min(x)) for x in iterates) >= 0.0 and np.min(res.x) >= 0.0


def test_a_step_landing_exactly_on_a_bound_stays_inside():
    """``log(x0 / 1e-10) + 1`` from 1e-10: the full step lands exactly on 0."""
    def f(x):
        return np.array([np.log(x[0] / 1e-10) + 1.0, x[1] - 1.0]), diags_array([1.0 / x[0], 1.0]).tocsc()

    res = newton(f, np.array([1e-10, 1.0]), bounds=(0.0, np.inf), maxfev=1)
    assert res.x[0] > 0.0 and np.all(np.isfinite(f(res.x)[0]))

    # at a nonzero bound eps * distance is below the float spacing: still inside
    def g(x):
        return np.log((x - 1.0) / 2.0**-30) + 1.0, diags_array(1.0 / (x - 1.0)).tocsc()

    for lower, x0 in ((1.0, 1.0 + 2.0**-30),):
        res = newton(g, np.array([x0]), bounds=(lower, np.inf), maxfev=1)
        assert res.x[0] > lower and np.all(np.isfinite(g(res.x)[0]))
    # and the same at an upper bound
    res = newton(lambda x: (-g(2.0 - x)[0], g(2.0 - x)[1]), np.array([1.0 - 2.0**-30]),
                 bounds=(-np.inf, 1.0), maxfev=1)
    assert res.x[0] < 1.0


def test_a_round_off_crossing_near_a_huge_bound_does_not_overflow():
    # the root lies 1e298 below the bound 1e308: a crossing of 1e-9 of the distance
    def f(x):
        return x - (1e308 - 1e298), diags_array([1.0]).tocsc()

    with np.errstate(over="ignore"):    # the reported backward error overflows here
        res = newton(f, np.array([1.1e308]), bounds=(1e308, np.inf), maxfev=1,
                     backward_tol=None)
    # the step is taken and reflected inside; overflow made it non-finite and refused
    assert 1e308 <= res.x[0] < 1.01e308


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


# ---------------------------------------------------------------------------
# Pseudo-time stepping and continuation
# ---------------------------------------------------------------------------

def test_pseudo_transient_reaches_the_steady_state_newton_misses():
    from pymrm import pseudo_transient
    fun, shape = _weisz_hicks(phi=0.8, gamma=30.0, beta=0.5)
    x0 = np.full(shape, 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        guarded = newton(fun, x0, damping="guarded")
        monotone = newton(fun, x0, damping="monotone")
        march = pseudo_transient(fun, x0, dt=1e-2)
    assert not guarded.success and not monotone.success
    assert march.success and march.steady.status == 0 and march.x.min() < 1e-3
    assert 0 < march.nsteps <= 20


def test_continuation_follows_a_branch_and_stops_at_its_fold():
    from pymrm import continuation

    def fun(c, phi):
        return _weisz_hicks(phi=phi, gamma=20.0, beta=0.6)[0](c)
    _, shape = _weisz_hicks(phi=0.1, gamma=20.0, beta=0.6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        up = continuation(fun, np.ones(shape), np.linspace(0.1, 0.6, 11))
        down = continuation(fun, np.zeros(shape), np.linspace(0.6, 0.1, 11))
    assert not up.success and 0.55 < up.last_param < 0.6     # ignition fold
    assert min(x.min() for x in up.xs) > 0.85                # never left the cold branch
    assert not down.success and 0.25 < down.last_param < 0.35  # extinction fold
    both = [p for p in up.params if p in down.params]
    assert both                                              # the multiplicity window
    for p in both:
        assert up.xs[up.params.index(p)].min() > 0.85
        assert down.xs[down.params.index(p)].min() < 0.05


def test_continuation_from_a_zero_solution_with_a_repeated_value():
    """Bratu, u'' + lam exp(u) = 0: from u = 0 at lam = 0 up to the fold at 3.514."""
    from pymrm import continuation
    n = 50
    h = 1.0 / (n + 1)
    lap = diags_array([np.ones(n - 1), -2.0 * np.ones(n), np.ones(n - 1)], offsets=[-1, 0, 1]) / h**2

    def fun(u, lam):
        return lap @ u + lam * np.exp(u), (lap + diags_array(lam * np.exp(u))).tocsc()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = continuation(fun, np.zeros(n), [0.0, 1.0, 1.0, 2.0, 3.0, 3.6])
    assert res.params == [0.0, 1.0, 1.0, 2.0, 3.0]
    assert not res.success and 3.4 < res.last_param < 3.514
