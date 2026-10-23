"""Operators on an axis with one cell, where each face depends on both boundary conditions."""
import numpy as np
import pytest

from pymrm import (
    construct_boundary_value_matrices,
    construct_convflux_upwind,
    construct_grad,
    interp_cntr_to_stagg_tvd,
    minmod,
)

ROBIN = ({"a": 1.3, "b": 0.7, "d": [[2.0, 1.0]]}, {"a": 0.4, "b": 1.1, "d": [[0.5, 3.0]]})
OUTFLOW = ({"a": 0.0, "b": 1.0, "d": [[2.0, 1.0]]}, {"outflow": True})


def _with_vectors(bc, ext):
    """Fold the external vectors into ``d``: the bc without shapes_d."""
    return tuple(b if "outflow" in b else {**b, "d": np.asarray(b["d"]) * e} for b, e in zip(bc, ext))


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("bc", [ROBIN, OUTFLOW], ids=["robin", "outflow"])
@pytest.mark.parametrize("operator", ["grad", "convflux"])
def test_shapes_d_matrices_match_folded_d(n, bc, operator):
    if operator == "grad" and "outflow" in bc[1]:
        pytest.skip("outflow is a convection boundary")
    rng = np.random.default_rng(n)
    x_f = np.cumsum(np.r_[0.0, rng.random(n) + 0.5])
    shape, shape_d = (4, n, 2), (4, 1, 2)
    ext = (rng.random(shape_d), rng.random(shape_d))
    c = rng.random(shape).ravel()
    v = rng.random((4, n + 1, 2)) + 0.1

    def build(**kwargs):
        if operator == "grad":
            return construct_grad(shape, x_f, axis=1, **kwargs)
        return construct_convflux_upwind(shape, x_f, v=v, axis=1, **kwargs)

    matrix, matrix_bc = build(bc=_with_vectors(bc, ext))
    expected = matrix @ c + matrix_bc.toarray().ravel()
    matrix, bc_left, bc_right = build(bc=bc, shapes_d=(shape_d, shape_d))
    got = matrix @ c + bc_left @ ext[0].ravel() + bc_right @ ext[1].ravel()
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-13)


def test_tvd_one_cell_returns_face_shape():
    values, delta = interp_cntr_to_stagg_tvd(
        np.ones((1, 3)), np.array([0.0, 1.0]), None,
        ({"a": 0, "b": 1, "d": 2}, {"a": 1, "b": 0, "d": 0}), 1.0, minmod, axis=0,
    )
    assert values.shape == (2, 3) and delta.shape == (2, 3)
    np.testing.assert_allclose(values[0], 2.0)


def test_boundary_value_matrices_reject_one_cell():
    with pytest.raises(ValueError, match="at least two cells"):
        construct_boundary_value_matrices(
            (1, 3), np.array([0.0, 1.0]), bc={"a": 1, "b": 1, "d": 1}, axis=0
        )
