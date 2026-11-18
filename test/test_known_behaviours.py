"""Tests that pin the known behaviours listed in the c2 maintainer note (code-health campaign c3).

A pin asserts today's behaviour. An xfail asserts the correct behaviour and fails today, so
that a fix flips it.
"""
import numpy as np
import pytest
from scipy.sparse import csc_array
from scipy.sparse.linalg import splu

from pymrm import NumJac
from pymrm.solve import newton


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
