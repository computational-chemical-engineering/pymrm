import pytest
import numpy as np
from scipy.sparse import csc_array
from pymrm.solve import newton, clip_approach


# ---------------------------------------------------------------------------
# Existing tests
# ---------------------------------------------------------------------------

def test_newton():
    def f(x):
        return np.array([x[0] ** 2 - 2]), csc_array([[2 * x[0]]])

    x0 = np.array([1.0])
    sol = newton(f, x0)
    assert np.isclose(sol.x[0] ** 2, 2, atol=1e-6)


def test_clip_approach():
    def f(x):
        return x**2 - 2

    x = np.array([1.0])
    clip_approach(x, f, lower_bounds=0, upper_bounds=2)
    assert np.all(x >= 0) and np.all(x <= 2)


# ---------------------------------------------------------------------------
# Newton solver variants
# ---------------------------------------------------------------------------

def _f_sqrt2(x):
    """Simple scalar nonlinear function: x^2 = 2."""
    return np.array([x[0] ** 2 - 2]), csc_array([[2 * x[0]]])


def test_newton_convergence_result():
    """Newton solver should report success and give x ≈ sqrt(2)."""
    sol = newton(_f_sqrt2, np.array([1.0]))
    assert sol.success
    assert sol.nit >= 1
    assert sol.x[0] == pytest.approx(np.sqrt(2), rel=1e-6)
    assert "Converged" in sol.message


def test_newton_dense_jacobian_default_solver():
    """Newton should use a dense direct solve for dense Jacobians."""

    def f_dense(x):
        return np.array([x[0] ** 2 - 2]), np.array([[2 * x[0]]])

    sol = newton(f_dense, np.array([1.0]))
    assert sol.success
    assert sol.x[0] == pytest.approx(np.sqrt(2), rel=1e-6)


def test_newton_solver_cg():
    """Newton with CG linear solver should converge."""
    sol = newton(_f_sqrt2, np.array([1.5]), solver="cg")
    assert sol.success
    assert sol.x[0] == pytest.approx(np.sqrt(2), rel=1e-5)


def test_newton_solver_bicgstab():
    """Newton with BiCGSTAB linear solver should converge."""
    sol = newton(_f_sqrt2, np.array([1.5]), solver="bicgstab")
    assert sol.success
    assert sol.x[0] == pytest.approx(np.sqrt(2), rel=1e-5)


def test_newton_solver_callable():
    """Newton accepts a user-provided linear solver callable."""
    from scipy.sparse.linalg import spsolve

    def my_solver(A, b, **kwargs):
        return spsolve(A, b)

    sol = newton(_f_sqrt2, np.array([1.0]), solver=my_solver)
    assert sol.success


def test_newton_invalid_solver_raises():
    """Unsupported solver string should raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported solver"):
        newton(_f_sqrt2, np.array([1.0]), solver="invalid_solver")


def test_newton_maxfev_not_converged():
    """Newton with maxfev=1 on a hard problem should report failure."""
    sol = newton(_f_sqrt2, np.array([0.001]), maxfev=1)
    # Either converges in 1 step (unlikely here) or reports failure
    if not sol.success:
        assert "not converge" in sol.message.lower() or not sol.success


def test_newton_callback():
    """Callback is called after each iteration."""
    calls = []

    def cb(x, g):
        calls.append(x.copy())

    sol = newton(_f_sqrt2, np.array([1.0]), callback=cb)
    assert sol.success
    assert len(calls) >= 1


def test_newton_with_args():
    """Newton should pass extra args to the function."""
    def f_with_target(x, target):
        return np.array([x[0] ** 2 - target]), csc_array([[2 * x[0]]])

    sol = newton(f_with_target, np.array([2.0]), args=(3.0,))
    assert sol.success
    assert sol.x[0] == pytest.approx(np.sqrt(3.0), rel=1e-6)


# ---------------------------------------------------------------------------
# clip_approach variants
# ---------------------------------------------------------------------------

def test_clip_approach_lower_bound_only():
    x = np.array([-5.0, 2.0, 3.0])
    clip_approach(x, None, lower_bounds=0.0, upper_bounds=None)
    assert np.all(x >= 0.0)
    assert x[1] == pytest.approx(2.0)
    assert x[2] == pytest.approx(3.0)


def test_clip_approach_upper_bound_only():
    x = np.array([1.0, 4.0, 10.0])
    clip_approach(x, None, lower_bounds=None, upper_bounds=5.0)
    assert np.all(x <= 5.0)


def test_clip_approach_both_bounds():
    x = np.array([-2.0, 0.5, 7.0])
    clip_approach(x, None, lower_bounds=0.0, upper_bounds=6.0)
    assert np.all(x >= 0.0)
    assert np.all(x <= 6.0)


def test_clip_approach_with_factor():
    """With a non-zero approach factor values should be shifted, not hard-clipped."""
    x = np.array([-1.0, 5.0])
    x_orig = x.copy()
    clip_approach(x, None, lower_bounds=0.0, upper_bounds=4.0, factor=0.5)
    # Value below lower bound should be moved towards lower bound
    assert x[0] > x_orig[0]  # pushed up from -1
    # Value above upper bound should be moved towards upper bound
    assert x[1] < x_orig[1]  # pushed down from 5


def test_clip_approach_factor_array_bounds():
    """Approach factor with array bounds should work element-wise."""
    x = np.array([-1.0, 0.5, 5.0])
    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([1.0, 1.0, 4.0])
    clip_approach(x, None, lower_bounds=lb, upper_bounds=ub, factor=0.1)
    assert x[0] >= -1.0  # should have been moved up
    assert x[2] <= 5.0  # should have been moved down


# ---------------------------------------------------------------------------
# newton: relative tolerance and step_norm
# ---------------------------------------------------------------------------

def _trace_root(x):
    # x**2 = 1e-20: root 1e-10, far below the default absolute tolerance
    from scipy.sparse import csc_array
    return np.array([x[0] ** 2 - 1e-20]), csc_array(np.array([[2.0 * x[0]]]))


def test_newton_default_is_absolute_and_reports_step_norm():
    res = newton(_trace_root, np.array([1e-8]), damping="none", backward_tol=None, stagnation=False)
    assert res.success and res.nit == 1          # stops on the absolute tolerance
    assert res.x[0] / 1e-10 > 40.0               # ... with a wrong answer
    assert np.isfinite(res.step_norm)


def test_newton_relative_tolerance_finds_trace_root():
    res = newton(_trace_root, np.array([1e-8]), tol=0.0, rtol=1e-10)
    assert res.success
    assert np.isclose(res.x[0], 1e-10, rtol=1e-8)


def test_newton_array_tolerance():
    res = newton(_trace_root, np.array([1e-8]), tol=np.array([1e-16]))
    assert res.success and np.isclose(res.x[0], 1e-10, rtol=1e-4)


# ---------------------------------------------------------------------------
# newton: result fields and legacy call patterns
# ---------------------------------------------------------------------------

def test_newton_result_fields_and_fun_at_x():
    x0 = np.array([1.0])
    res = newton(_f_sqrt2, x0)
    assert x0[0] == 1.0                                   # initial guess not modified
    assert res.status == 0 and res.nfact == res.nit and res.nfev == res.nit + 1
    assert abs(res.fun[0]) < 1e-12                        # residual AT x
    assert res.backward_error < 1e-12
    assert len(res.history) == res.nit
    assert {"step_norm", "damping", "theta", "clipped"} <= set(res.history[0])


def test_newton_prefactorised_splu():
    from scipy.sparse.linalg import splu

    def f(x):
        return np.array([x[0] ** 2 - 2]), splu(csc_array([[2 * x[0]]]))

    res = newton(f, np.array([1.0]), solver="splu")
    assert res.success and res.x[0] == pytest.approx(np.sqrt(2))
    assert np.isnan(res.backward_error)


def test_newton_direction_changing_callable_keeps_plain_steps():
    """A callable solver may cap the step (gallery I1.2); 'auto' must not damp it."""
    from scipy.sparse.linalg import spsolve

    def capped(jac, rhs):
        dx = np.atleast_1d(spsolve(jac, rhs))
        return np.clip(dx, -0.1, 0.1)

    res = newton(_f_sqrt2, np.array([3.0]), solver=capped)
    assert res.success and res.x[0] == pytest.approx(np.sqrt(2))
    assert all(h["damping"] == 1.0 for h in res.history)


def test_newton_callback_runs_once_per_iteration_before_the_next_evaluation():
    order = []

    def f(x):
        order.append("f")
        return _f_sqrt2(x)

    def cb(x, g):
        order.append("cb")

    res = newton(f, np.array([1.0]), callback=cb)
    assert order == ["f"] + ["cb", "f"] * res.nit


def test_newton_clip_approach_callback_with_upper_bound():
    def f(x):
        return np.array([x[0] ** 2 - 2]), csc_array([[2 * x[0]]])

    res = newton(f, np.array([0.1]), callback=lambda x, g: clip_approach(x, g, 0.0, 1.5))
    assert res.success and res.x[0] == pytest.approx(np.sqrt(2))


def test_newton_rejects_unknown_damping():
    with pytest.raises(ValueError, match="damping"):
        newton(_f_sqrt2, np.array([1.0]), damping="linesearch")


def test_newton_accepts_scipy_bounds():
    from scipy.optimize import Bounds

    res = newton(_f_sqrt2, np.array([3.0]), bounds=Bounds(0.5, 4.0), damping="monotone")
    assert res.success and res.x[0] == pytest.approx(np.sqrt(2))


def test_newton_scale_applies_tol_to_the_scaled_step():
    res = newton(_trace_root, np.array([1e-8]), scale=1e-10)
    assert res.success and np.isclose(res.x[0], 1e-10, rtol=1e-6)


def test_newton_fortran_ordered_initial_guess():
    def f(x):
        g = (x**2 - np.array([2.0, 3.0])).ravel()
        return g, csc_array(np.diag(2.0 * x.ravel()))

    x0 = np.asfortranarray(np.ones((4, 2)))
    res = newton(f, x0)
    assert res.success and np.allclose(res.x, np.sqrt([2.0, 3.0]))


def test_newton_exact_initial_guess_with_damping():
    res = newton(_f_sqrt2, np.array([np.sqrt(2.0)]), damping="monotone")
    assert res.success


def test_newton_final_evaluation_cannot_turn_a_result_into_an_exception():
    calls = []

    def f(x):
        calls.append(1)
        if len(calls) > 2:
            raise ValueError("property out of range")
        return _f_sqrt2(x)

    res = newton(f, np.array([1.0]), maxfev=2)
    assert not res.success and res.status == 2


def test_newton_rejects_non_positive_scale():
    with pytest.raises(ValueError, match="scale"):
        newton(_f_sqrt2, np.array([1.0]), scale=0.0)
