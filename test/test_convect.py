import pytest
import numpy as np
from scipy.sparse import issparse
from pymrm.convect import (
    construct_convflux_upwind,
    construct_convflux_upwind_int,
    construct_convflux_bc,
    upwind,
    minmod,
    osher,
    clam,
    muscl,
    smart,
    stoic,
    vanleer,
)
from pymrm.grid import generate_grid


@pytest.fixture(params=["csc", "csr"])
def sparse_format(request):
    return request.param


# ---------------------------------------------------------------------------
# Basic construction (existing tests)
# ---------------------------------------------------------------------------

def test_construct_convflux_upwind(sparse_format):
    shape = (10,)
    x_f = np.linspace(0, 1, shape[0] + 1)
    conv_matrix, conv_bc = construct_convflux_upwind(shape, x_f, format=sparse_format)
    assert conv_matrix.shape[0] > 0
    assert conv_bc.shape[0] > 0


def test_construct_convflux_upwind_int(sparse_format):
    shape = (10,)
    v = np.ones(11)
    conv_matrix = construct_convflux_upwind_int(shape, v, format=sparse_format)
    assert conv_matrix.shape[0] > 0


def test_construct_convflux_bc(sparse_format):
    shape = (10,)
    x_f = np.linspace(0, 1, shape[0] + 1)
    v = np.ones(11)
    bc = ({"a": 0, "b": 1, "d": 1}, {"a": 1, "b": 0, "d": 0})
    result = construct_convflux_bc(shape, x_f, bc=bc, v=v, format=sparse_format)
    if isinstance(result, tuple):
        conv_matrix, conv_bc = result[0], result[1]
        assert conv_matrix.shape[0] > 0
        assert conv_bc.shape[0] > 0
    else:
        assert result.shape[0] > 0


def test_tvd_limiters():
    c = np.linspace(-1, 2, 10)
    x_c = 0.5
    x_d = 0.75
    for limiter in [upwind, minmod, osher, clam, muscl, smart, stoic, vanleer]:
        result = limiter(c, x_c, x_d)
        assert result.shape == c.shape


# ---------------------------------------------------------------------------
# Shape checks
# ---------------------------------------------------------------------------

def test_convflux_upwind_shapes(sparse_format):
    n = 8
    shape = (n,)
    x_f = np.linspace(0, 1, n + 1)
    bc = ({"a": 0, "b": 1, "d": 0.0}, {"a": 0, "b": 1, "d": 1.0})
    v = 1.0
    conv_matrix, conv_bc = construct_convflux_upwind(shape, x_f, bc=bc, v=v,
                                                     format=sparse_format)
    assert conv_matrix.shape == (n + 1, n)
    assert conv_bc.shape[0] == n + 1
    assert issparse(conv_matrix)
    assert issparse(conv_bc)


# ---------------------------------------------------------------------------
# Positive vs. negative velocity (upwind direction)
# ---------------------------------------------------------------------------

def test_convflux_upwind_positive_velocity(sparse_format):
    """Positive velocity: face flux comes from left (upstream) cell."""
    n = 5
    shape = (n,)
    x_f = np.linspace(0, 1, n + 1)
    v = 1.0  # positive, all upwind from left
    conv_int = construct_convflux_upwind_int(shape, v, format=sparse_format)
    c = np.arange(1.0, n + 1)
    face_fluxes = (conv_int @ c)
    # All internal face fluxes should equal v * c[j] for j=0..n-2 (upwind from left)
    for j in range(1, n):  # internal faces 1..n-1
        assert face_fluxes[j] == pytest.approx(v * c[j - 1], abs=1e-12)


def test_convflux_upwind_negative_velocity(sparse_format):
    """Negative velocity: face flux comes from right (upstream) cell."""
    n = 5
    shape = (n,)
    x_f = np.linspace(0, 1, n + 1)
    v = -1.0  # negative, upwind from right
    conv_int = construct_convflux_upwind_int(shape, v, format=sparse_format)
    c = np.arange(1.0, n + 1)
    face_fluxes = (conv_int @ c)
    # For v < 0, internal face j (between cells j-1 and j) should = v * c[j]
    for j in range(1, n):
        assert face_fluxes[j] == pytest.approx(v * c[j], abs=1e-12)


def test_convflux_upwind_zero_velocity(sparse_format):
    """Zero velocity should result in zero flux."""
    n = 6
    shape = (n,)
    x_f = np.linspace(0, 1, n + 1)
    conv_int = construct_convflux_upwind_int(shape, v=0.0, format=sparse_format)
    c = np.ones(n)
    face_fluxes = conv_int @ c
    np.testing.assert_allclose(face_fluxes, 0.0, atol=1e-14)


def test_convflux_upwind_negative_velocity_with_bc(sparse_format):
    """Negative velocity with inlet at right boundary."""
    n = 6
    shape = (n,)
    x_f = np.linspace(0, 1, n + 1)
    v_val = -2.0
    # For negative flow, c enters from the right (c=1), exits at left
    bc = ({"a": 1, "b": 0, "d": 0}, {"a": 0, "b": 1, "d": 1.0})
    v = np.full(n + 1, v_val)
    conv_matrix, conv_bc = construct_convflux_upwind(shape, x_f, bc=bc, v=v,
                                                     format=sparse_format)
    assert issparse(conv_matrix)
    assert issparse(conv_bc)


# ---------------------------------------------------------------------------
# Single-cell special case
# ---------------------------------------------------------------------------

def test_construct_convflux_bc_single_cell(sparse_format):
    """Single-cell domain triggers the special case in construct_convflux_bc."""
    shape = (1,)
    x_f = np.array([0.0, 1.0])
    bc = ({"a": 0, "b": 1, "d": 1.0}, {"a": 0, "b": 1, "d": 2.0})
    v = 1.0
    conv_matrix, conv_bc = construct_convflux_upwind(shape, x_f, bc=bc, v=v,
                                                     format=sparse_format)
    assert issparse(conv_matrix)
    assert issparse(conv_bc)


def test_construct_convflux_bc_single_cell_negative_v(sparse_format):
    """Single-cell domain with negative velocity."""
    shape = (1,)
    x_f = np.array([0.0, 1.0])
    bc = ({"a": 0, "b": 1, "d": 1.0}, {"a": 0, "b": 1, "d": 2.0})
    v = -1.0
    conv_matrix, conv_bc = construct_convflux_upwind(shape, x_f, bc=bc, v=v,
                                                     format=sparse_format)
    assert issparse(conv_matrix)


# ---------------------------------------------------------------------------
# shapes_d argument
# ---------------------------------------------------------------------------

def test_construct_convflux_with_shapes_d(sparse_format):
    """shapes_d argument enables coupled BC decomposition."""
    n = 6
    shape = (n,)
    x_f = np.linspace(0, 1, n + 1)
    bc = ({"a": 0, "b": 1, "d": 0.0}, {"a": 0, "b": 1, "d": 1.0})
    result = construct_convflux_upwind(shape, x_f, bc=bc, shapes_d=((1,), (1,)),
                                       format=sparse_format)
    # Should return 3 values: conv_matrix, conv_bc_0, conv_bc_1
    assert len(result) == 3


# ---------------------------------------------------------------------------
# 2-D domain
# ---------------------------------------------------------------------------

def test_convflux_upwind_2d_shape(sparse_format):
    """2D convection operator should have the right shape."""
    shape = (5, 4)
    n_x, n_y = shape
    x_f = np.linspace(0, 1, n_x + 1)
    bc = ({"a": 0, "b": 1, "d": 0.0}, {"a": 0, "b": 1, "d": 1.0})
    conv, conv_bc = construct_convflux_upwind(shape, x_f, bc=bc, axis=0,
                                              format=sparse_format)
    assert conv.shape == ((n_x + 1) * n_y, n_x * n_y)


# ---------------------------------------------------------------------------
# TVD limiters – boundary properties
# ---------------------------------------------------------------------------

def test_tvd_limiters_upwind_region():
    """Most limiters should return 0 for c_norm ≤ 0 (oscillation region).

    Note: stoic has a distinct functional form and can return positive values
    for negative normalized concentrations, which is its documented behaviour.
    """
    c_norm = np.array([-0.5, -0.1, 0.0])
    x_c = 0.4
    x_d = 0.6
    # These limiters are known to return 0 in the upstream / non-monotone region
    for lim in [minmod, osher, clam, muscl, smart, vanleer]:
        result = lim(c_norm, x_c, x_d)
        np.testing.assert_array_less(result, 1e-10 + np.zeros_like(result))


def test_tvd_limiters_output_shape_2d():
    """TVD limiters should handle 2D arrays (matching the caller shape)."""
    c = np.random.rand(5, 3)
    x_c = 0.4
    x_d = 0.6
    for lim in [upwind, minmod, osher, clam, muscl, smart, stoic, vanleer]:
        result = lim(c, x_c, x_d)
        assert result.shape == c.shape


# ---------------------------------------------------------------------------
# Direct construct_convflux_bc with scalar velocity (covers the scalar branch)
# ---------------------------------------------------------------------------

def test_construct_convflux_bc_scalar_v(sparse_format):
    """Calling construct_convflux_bc directly with scalar v covers scalar path."""
    n = 6
    shape = (n,)
    x_f = np.linspace(0, 1, n + 1)
    x_c = 0.5 * (x_f[:-1] + x_f[1:])
    bc = ({"a": 0, "b": 1, "d": 0.0}, {"a": 0, "b": 1, "d": 1.0})
    v_scalar = 2.5  # float scalar
    conv_matrix, conv_bc = construct_convflux_bc(shape, x_f, x_c=x_c, bc=bc,
                                                 v=v_scalar, format=sparse_format)
    assert issparse(conv_matrix)
    assert issparse(conv_bc)


def test_construct_convflux_bc_scalar_v_negative(sparse_format):
    """Scalar negative v through construct_convflux_bc directly."""
    n = 6
    shape = (n,)
    x_f = np.linspace(0, 1, n + 1)
    x_c = 0.5 * (x_f[:-1] + x_f[1:])
    bc = ({"a": 0, "b": 1, "d": 0.0}, {"a": 0, "b": 1, "d": 1.0})
    conv_matrix, conv_bc = construct_convflux_bc(shape, x_f, x_c=x_c, bc=bc,
                                                 v=-1.0, format=sparse_format)
    assert issparse(conv_matrix)


def test_construct_convflux_upwind_int_scalar(sparse_format):
    """construct_convflux_upwind_int with scalar v (covers scalar path in int)."""
    n = 6
    shape = (n,)
    # v as integer scalar
    conv_matrix = construct_convflux_upwind_int(shape, v=2, format=sparse_format)
    assert issparse(conv_matrix)


def test_construct_convflux_upwind_int_scalar_shape(sparse_format):
    """construct_convflux_upwind with integer shape."""
    x_f = np.linspace(0, 1, 9)
    conv_matrix, conv_bc = construct_convflux_upwind(8, x_f, format=sparse_format)
    assert conv_matrix.shape[1] == 8
