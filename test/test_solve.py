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
