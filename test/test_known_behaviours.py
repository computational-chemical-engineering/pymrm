"""Tests that pin the known behaviours listed in the c2 maintainer note (code-health campaign c3).

A pin asserts today's behaviour. An xfail asserts the correct behaviour and fails today, so
that a fix flips it.
"""
import re
import warnings

import numpy as np
import pytest
from scipy.sparse import csc_array, dia_array, random as sparse_random
from scipy.sparse.linalg import splu

import pymrm.solve
from pymrm import NumJac, continuation, pseudo_transient
from pymrm.helpers import _csc_abs_matvec
from pymrm.solve import _Bounds, newton


@pytest.mark.xfail(strict=True, raises=TypeError, reason="maintainer-note-c2 item 1: NumJac int shape")
def test_numjac_int_shape_equals_tuple_shape():
    """maintainer-note-c2 item 1: NumJac(shape=100) builds the same Jacobian as NumJac(shape=(100,))."""
    c = np.linspace(1.0, 2.0, 100)
    with pytest.warns(UserWarning, match="dense"):
        _, jac_tuple = NumJac(shape=(100,))(lambda v: v**2, c)
    with pytest.warns(UserWarning, match="dense"):
        _, jac_int = NumJac(shape=100)(lambda v: v**2, c)
    np.testing.assert_array_equal(jac_int.toarray(), jac_tuple.toarray())


def _coupled_system(x):
    """Three coupled exponential, cubic and hyperbolic equations with an exact Jacobian."""
    g = np.array([np.exp(x[0]) + x[1] - 3, x[0] * x[1]**3 - 2 + x[2], np.sinh(x[2]) + x[0] - 1])
    jac = np.array([[np.exp(x[0]), 1, 0], [x[1]**3, 3 * x[0] * x[1]**2, 1], [1, 0, np.cosh(x[2])]])
    return g, csc_array(jac)


class _LuSolver:
    """A user linear solver (factorize protocol); with ``reuse`` every solve returns one shared buffer."""

    def __init__(self, reuse):
        self.reuse = reuse
        self.buffer = np.empty(3)

    def factorize(self, jac):
        lu = splu(csc_array(jac))
        if not self.reuse:
            return lu
        buffer = self.buffer

        class _Reusing:
            @staticmethod
            def solve(rhs):
                buffer[:] = lu.solve(rhs)
                return buffer
        return _Reusing()


@pytest.mark.xfail(strict=True, reason="maintainer-note-c2 item 2: dx aliases a reused solver buffer")
def test_newton_user_solver_reusing_its_buffer_matches_fresh_output():
    """maintainer-note-c2 item 2: a user solver that reuses its output buffer gives the same iteration."""
    x0 = np.array([-2.0, 2.0, 2.0])
    fresh = newton(_coupled_system, x0, solver=_LuSolver(False), damping="monotone")
    reused = newton(_coupled_system, x0, solver=_LuSolver(True), damping="monotone")
    assert fresh.status == reused.status == 0
    assert (reused.nit, reused.nfev) == (fresh.nit, fresh.nfev)
    np.testing.assert_array_equal(reused.x, fresh.x)


def test_newton_empty_guess_raises_value_error():
    """maintainer-note-c2 item 3: newton on n = 0 raises ValueError.

    Pins current behaviour; a deliberate change updates this test.
    """
    with pytest.raises(ValueError):
        newton(lambda x: (np.zeros(0), csc_array((0, 0))), np.zeros(0))


@pytest.mark.parametrize("dense", [False, True])
def test_newton_accepts_a_bool_sparse_residual(dense):
    """maintainer-note-c2 item 3: a bool sparse residual is accepted and gives the float residual's x.

    Pins current behaviour; a deliberate change updates this test.
    """
    def make(dtype):
        def function(x):
            g = csc_array((x > 0.5).astype(float).reshape(-1, 1)).astype(dtype)
            return g, (np.eye(2) if dense else csc_array(np.eye(2)))
        return function

    x0 = np.array([1.2, 0.3])
    as_float = newton(make(float), x0)
    as_bool = newton(make(bool), x0)
    assert as_float.status == as_bool.status == 0
    assert as_bool.x.tobytes() == as_float.x.tobytes()


_TRIDIAGONAL = np.diag([3.0, 3.0, 3.0]) + np.diag([1.0, 1.0], 1) + np.diag([1.0, 1.0], -1)


def _cubic_chain(matrix_type):
    """A tridiagonal linear term plus a cubic, with its Jacobian in the sparse type ``matrix_type``."""
    def function(x):
        return _TRIDIAGONAL @ x + x**3 - np.array([1.0, 2.0, 3.0]), matrix_type(_TRIDIAGONAL + np.diag(3 * x**2))
    return function


@pytest.mark.xfail(strict=True, raises=AttributeError, reason="maintainer-note-c2 item 4: dia_array Jacobian")
@pytest.mark.parametrize("damping", ["monotone", "armijo"])
def test_newton_dia_array_jacobian_matches_csc(damping):
    """maintainer-note-c2 item 4: a dia_array Jacobian under damping converges to the csc_array result."""
    x0 = np.ones(3)
    expected = newton(_cubic_chain(csc_array), x0, damping=damping)
    result = newton(_cubic_chain(dia_array), x0, damping=damping)
    assert expected.status == result.status == 0
    np.testing.assert_allclose(result.x, expected.x, rtol=1e-12)


def test_newton_dense_splu_key_raises_without_damping_and_works_when_guarded():
    """maintainer-note-c2 item 5: a splu key in lin_solver_kwargs on a dense Jacobian.

    With damping="none" the key reaches the dense solver and raises TypeError, with damping="guarded" the
    solve succeeds. Neither case warns. Pins current behaviour; a deliberate change updates this test.
    """
    def function(x):
        return _TRIDIAGONAL @ x + x**3 - np.array([1.0, 2.0, 3.0]), _TRIDIAGONAL + np.diag(3 * x**2)

    kwargs = {"permc_spec": "NATURAL"}
    with pytest.raises(TypeError):
        newton(function, np.ones(3), damping="none", lin_solver_kwargs=kwargs)
    result = newton(function, np.ones(3), damping="guarded", lin_solver_kwargs=kwargs)
    assert result.status == 0


def _raising_after_the_first_call():
    """sqrt(2) residual that evaluates once and raises ValueError at every later point."""
    calls = []

    def function(x):
        calls.append(1)
        if len(calls) > 1:
            raise ValueError("bad trial point")
        return np.array([x[0]**2 - 2]), csc_array([[2 * x[0]]])
    return function


def test_newton_monotone_error_at_the_first_trial_point_depends_on_maxfev():
    """maintainer-note-c2 item 6: a ValueError at the first trial point gives status 2 or raises.

    With maxfev=1 that iteration is the last: status 2 and a message naming ValueError. With
    maxfev=2 the ValueError propagates. Pins current behaviour; a deliberate change updates this test.
    """
    result = newton(_raising_after_the_first_call(), np.ones(1), damping="monotone", maxfev=1)
    assert result.status == 2
    assert "ValueError" in result.message
    with pytest.raises(ValueError):
        newton(_raising_after_the_first_call(), np.ones(1), damping="monotone", maxfev=2)


def test_bounds_commit_marks_a_component_that_project_did_not_hold():
    """maintainer-note-c2 item 7: _Bounds.commit finds held components by float equality.

    For x = 1, lower = 0 and dx = 1.5 the step is limited to lam = 0.66 and lands at 0.01: project holds
    no component, yet commit marks it held. Pins current behaviour; a deliberate change updates this test.
    """
    bounds = _Bounds(np.array([0.0]), None, 0.99)
    x, dx = np.array([1.0]), np.array([1.5])
    lam = bounds.max_step(x, dx)
    assert lam == pytest.approx(0.66)
    trial = x - lam * dx
    assert bounds.project(trial, x, dx) == 0
    bounds.commit(x, trial, dx)
    assert bounds.held.tolist() == [True]


def test_pseudo_transient_failure_message_reports_a_dt_below_every_dt_tried(monkeypatch):
    """maintainer-note-c2 item 8: the failure message names a dt below the smallest dt tried.

    The march shrinks dt by 4 after each failed try and reports the shrunk value after the last one.
    Pins current behaviour; a deliberate change updates this test.
    """
    tried = []
    original = pymrm.solve._transient_residual

    def spy(function, mass_matrix, x_old, dt):
        tried.append(dt)
        return original(function, mass_matrix, x_old, dt)

    monkeypatch.setattr(pymrm.solve, "_transient_residual", spy)
    result = pseudo_transient(lambda x: (np.ones(1), csc_array([[1.0]])), np.ones(1), dt=1e-3,
                              newton_kwargs={"tol": 0.0, "maxfev": 2})
    assert not result.success
    reported = float(re.search(r"failed down to dt = (\S+)", result.message).group(1))
    assert len(tried) > 1
    assert reported < min(tried)
    assert min(tried) / reported == pytest.approx(4.0, rel=1e-2)


def test_continuation_first_value_failure_returns_the_failed_result_only():
    """maintainer-note-c2 item 9: when the first value fails, results = [first], params = [] and xs = [].

    Pins current behaviour; a deliberate change updates this test.
    """
    def function(x, p):
        return np.array([x[0]**2 + 1.0]), csc_array([[2 * x[0]]])

    result = continuation(function, np.ones(1), [0.0, 1.0, 2.0])
    assert not result.success
    assert len(result.results) == 1 and not result.results[0].success
    assert result.params == [] and result.xs == []
    assert result.last_param is None
    assert "no solution at the first value" in result.message


def test_newton_ignored_lin_solver_kwargs_warning_reports_the_callers_file():
    """maintainer-note-c2 item 10: the "ignored" warning for lin_solver_kwargs names the caller's file.

    Pins current behaviour; a deliberate change updates this test.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        newton(lambda x: (x - 1.0, csc_array([[1.0]])), np.zeros(1), lin_solver_kwargs={"unknown_key": 1})
    ignored = [w for w in caught if "is ignored" in str(w.message)]
    assert len(ignored) == 1
    assert ignored[0].category is RuntimeWarning
    assert ignored[0].filename == __file__


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("n", [7, 40])
def test_csc_abs_matvec_equals_abs_matmul_bitwise(index_dtype, n):
    """maintainer-note-c2 item 12: helpers._csc_abs_matvec(jac, v) equals abs(jac) @ v bitwise.

    The helper calls a private scipy kernel (scipy >= 1.13). Pins current behaviour; a deliberate change
    updates this test.
    """
    rng = np.random.default_rng(12)
    jac = csc_array(sparse_random(n, n, density=0.3, random_state=1, format="csc"))
    jac.data = rng.standard_normal(jac.data.size)
    jac.indices = jac.indices.astype(index_dtype)
    jac.indptr = jac.indptr.astype(index_dtype)
    v = rng.standard_normal(n)
    result = _csc_abs_matvec(jac, v)
    assert jac.indices.dtype == index_dtype
    assert result.tobytes() == (abs(jac) @ v).tobytes()
