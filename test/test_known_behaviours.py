"""Tests that pin the known behaviours listed in the c2 maintainer note (code-health campaign c3).

A pin asserts today's behaviour. An xfail asserts the correct behaviour and fails today, so
that a fix flips it.
"""
import numpy as np
import pytest

from pymrm import NumJac


@pytest.mark.xfail(strict=True, raises=TypeError, reason="maintainer-note-c2 item 1: NumJac int shape")
def test_numjac_int_shape_equals_tuple_shape():
    """maintainer-note-c2 item 1: NumJac(shape=100) builds the same Jacobian as NumJac(shape=(100,))."""
    c = np.linspace(1.0, 2.0, 100)
    with pytest.warns(UserWarning, match="dense"):
        _, jac_tuple = NumJac(shape=(100,))(lambda v: v**2, c)
    with pytest.warns(UserWarning, match="dense"):
        _, jac_int = NumJac(shape=100)(lambda v: v**2, c)
    np.testing.assert_array_equal(jac_int.toarray(), jac_tuple.toarray())
