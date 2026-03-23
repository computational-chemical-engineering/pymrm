import pytest
import numpy as np
from pymrm.interpolate import (
    interp_stagg_to_cntr,
    interp_cntr_to_stagg,
    interp_cntr_to_stagg_tvd,
    create_staggered_array,
    compute_boundary_values,
    construct_boundary_value_matrices,
)
from pymrm.convect import upwind, minmod, vanleer
from pymrm.grid import generate_grid


# ---------------------------------------------------------------------------
# Existing tests
# ---------------------------------------------------------------------------

def test_interp_stagg_to_cntr():
    x_f = np.linspace(0.0, 1.0, 11)
    arr = 10 * x_f + 1.0
    result = interp_stagg_to_cntr(arr, x_f)
    assert result.shape[0] == 10


def test_interp_cntr_to_stagg():
    arr = np.arange(10.0)
    x_f = np.linspace(0, 1, 11)
    result = interp_cntr_to_stagg(arr, x_f)
    assert result.shape[0] == 11


def test_interp_cntr_to_stagg_tvd():
    arr = np.arange(10.0)
    x_f = np.linspace(0, 1, 11)
    x_c = np.linspace(0.05, 0.95, 10)
    bc = ({"a": 0, "b": 1, "d": 1}, {"a": 1, "b": 0, "d": 0})
    v = 1.0
    result, _ = interp_cntr_to_stagg_tvd(arr, x_f, x_c, bc, v, upwind)
    assert result.shape[0] == 11


def test_create_staggered_array():
    arr = np.arange(10.0)
    shape = (10,)
    x_f = np.linspace(0, 1, 11)
    x_c = np.linspace(0.05, 0.95, 10)
    result = create_staggered_array(arr, shape, 0, x_f=x_f, x_c=x_c)
    assert result.shape[0] == 11


def test_compute_boundary_values():
    tol = 1e-12
    num_x = 2
    x_f = np.linspace(0.0, 1.0, num_x + 1)
    c = 0.5 * (x_f[1:] + x_f[:-1]).copy()

    # Left boundary condition
    bc_left = {"a": -1, "b": 1, "d": 1}
    boundary_value, boundary_grad = compute_boundary_values(
        c, x_f, bc=bc_left, bound_id=0
    )
    assert abs(boundary_value) < tol
    assert abs(boundary_grad - 1.0) < tol

    # Right boundary condition
    bc_right = {"a": 1, "b": 1, "d": 2}
    boundary_value, boundary_grad = compute_boundary_values(
        c, x_f, bc=bc_right, bound_id=1
    )
    assert abs(boundary_value - 1.0) < tol
    assert abs(boundary_grad - 1.0) < tol


def test_construct_boundary_value_matrices():
    tol = 1e-12
    num_x = 2
    x_f = np.linspace(0.0, 1.0, num_x + 1)
    c = 0.5 * (x_f[1:] + x_f[:-1]).copy()

    # Left boundary condition
    bc_left = {"a": -1, "b": 1, "d": 1}
    matrix, matrix_bc = construct_boundary_value_matrices(
        c.shape, x_f, bc=bc_left, bound_id=0
    )
    boundary_value = matrix @ c.reshape((-1, 1)) + matrix_bc
    assert np.allclose(boundary_value, 0.0, atol=tol)

    # Right boundary condition
    bc_right = {"a": 1, "b": 1, "d": 2}
    matrix, matrix_bc = construct_boundary_value_matrices(
        c.shape, x_f, bc=bc_right, bound_id=1
    )
    boundary_value = matrix @ c.reshape((-1, 1)) + matrix_bc
    assert np.allclose(boundary_value, 1.0, atol=tol)


# ---------------------------------------------------------------------------
# Accuracy on linear profiles
# ---------------------------------------------------------------------------

def test_interp_stagg_to_cntr_linear_exact():
    """Linear function on staggered grid should interpolate exactly to cell centers."""
    n = 8
    x_f = np.linspace(0.0, 1.0, n + 1)
    _, x_c = generate_grid(n, x_f, generate_x_c=True)
    f_f = 3.0 * x_f + 1.0
    result = interp_stagg_to_cntr(f_f, x_f, x_c)
    expected = 3.0 * x_c + 1.0
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_interp_cntr_to_stagg_linear_exact():
    """Interpolation of a linear profile from cell-center to face should be exact."""
    n = 8
    x_f = np.linspace(0.0, 1.0, n + 1)
    _, x_c = generate_grid(n, x_f, generate_x_c=True)
    c = 2.0 * x_c + 1.0
    result = interp_cntr_to_stagg(c, x_f, x_c)
    expected = 2.0 * x_f + 1.0
    # Interior faces should be exact for linear data
    np.testing.assert_allclose(result[1:-1], expected[1:-1], atol=1e-12)


# ---------------------------------------------------------------------------
# Single-cell domain
# ---------------------------------------------------------------------------

def test_interp_cntr_to_stagg_single_cell():
    """Single-cell domain: staggered values should equal the cell value."""
    x_f = np.array([0.0, 1.0])
    c = np.array([5.0])
    result = interp_cntr_to_stagg(c, x_f)
    assert result.shape[0] == 2
    np.testing.assert_allclose(result, 5.0, atol=1e-12)


def test_interp_cntr_to_stagg_tvd_single_cell():
    """TVD interpolation with a single-cell domain."""
    x_f = np.array([0.0, 1.0])
    x_c = np.array([0.5])
    c = np.array([3.0])
    bc = ({"a": 0, "b": 1, "d": 2.0}, {"a": 0, "b": 1, "d": 4.0})
    v = 1.0
    result, delta = interp_cntr_to_stagg_tvd(c, x_f, x_c, bc, v, upwind)
    # Single cell: result should contain two boundary-face values
    assert result.size == 2


# ---------------------------------------------------------------------------
# Multi-dimensional arrays
# ---------------------------------------------------------------------------

def test_interp_stagg_to_cntr_2d():
    """2D staggered array interpolation along axis=0."""
    n_x, n_y = 6, 4
    x_f = np.linspace(0.0, 1.0, n_x + 1)
    shape_f = (n_x + 1, n_y)
    arr = np.ones(shape_f)
    result = interp_stagg_to_cntr(arr, x_f, axis=0)
    assert result.shape == (n_x, n_y)


def test_interp_cntr_to_stagg_2d():
    """2D cell-centered array interpolation along axis=1."""
    n_x, n_y = 5, 6
    y_f = np.linspace(0.0, 2.0, n_y + 1)
    c = np.ones((n_x, n_y))
    result = interp_cntr_to_stagg(c, y_f, axis=1)
    assert result.shape == (n_x, n_y + 1)


def test_interp_cntr_to_stagg_tvd_with_minmod():
    """TVD interpolation with minmod limiter returns correct shape."""
    n = 12
    x_f = np.linspace(0, 1, n + 1)
    x_c = np.linspace(0.5 / n, 1 - 0.5 / n, n)
    c = np.sin(np.pi * x_c)
    bc = ({"a": 1, "b": 0, "d": 0.0}, {"a": 1, "b": 0, "d": 0.0})
    v = 1.0
    result, _ = interp_cntr_to_stagg_tvd(c, x_f, x_c, bc, v, minmod)
    assert result.shape[0] == n + 1


def test_interp_cntr_to_stagg_tvd_negative_velocity():
    """TVD interpolation with negative velocity."""
    n = 8
    x_f = np.linspace(0, 1, n + 1)
    x_c = 0.5 * (x_f[:-1] + x_f[1:])
    c = np.arange(n, dtype=float)
    bc = ({"a": 0, "b": 1, "d": 0.0}, {"a": 0, "b": 1, "d": 0.0})
    v = -1.0
    result, _ = interp_cntr_to_stagg_tvd(c, x_f, x_c, bc, v, vanleer)
    assert result.shape[0] == n + 1


# ---------------------------------------------------------------------------
# create_staggered_array with scalar input
# ---------------------------------------------------------------------------

def test_create_staggered_array_scalar():
    """Scalar velocity should broadcast to all face positions."""
    shape = (8,)
    x_f = np.linspace(0, 1, 9)
    x_c = 0.5 * (x_f[:-1] + x_f[1:])
    result = create_staggered_array(2.5, shape, 0, x_f=x_f, x_c=x_c)
    assert result.shape[0] == 9
    np.testing.assert_allclose(result, 2.5, atol=1e-12)


# ---------------------------------------------------------------------------
# construct_boundary_value_matrices with shape_d
# ---------------------------------------------------------------------------

def test_compute_boundary_values_bound_id_none():
    """bound_id=None returns all four boundary values."""
    n = 6
    x_f = np.linspace(0.0, 1.0, n + 1)
    _, x_c = generate_grid(n, x_f, generate_x_c=True)
    c = x_c.copy()
    bc = ({"a": 0, "b": 1, "d": 0.0}, {"a": 0, "b": 1, "d": 1.0})
    result = compute_boundary_values(c, x_f, bc=bc, bound_id=None)
    # Returns (val_0, grad_0, val_1, grad_1)
    assert len(result) == 4


def test_compute_boundary_values_single_cell_both_bounds():
    """Single-cell domain with bound_id=None triggers the single-cell path."""
    x_f = np.array([0.0, 1.0])
    x_c = np.array([0.5])
    c = np.array([2.0])
    bc = ({"a": 0, "b": 1, "d": 1.0}, {"a": 0, "b": 1, "d": 3.0})
    result = compute_boundary_values(c, x_f, bc=bc, bound_id=None)
    assert len(result) == 4


def test_construct_boundary_value_matrices_no_xc():
    """x_c=None: should auto-compute from x_f."""
    n = 4
    x_f = np.linspace(0.0, 1.0, n + 1)
    shape = (n,)
    bc = {"a": 0, "b": 1, "d": 1.0}
    # Should work without providing x_c
    matrix, mat_bc = construct_boundary_value_matrices(
        shape, x_f, x_c=None, bc=bc, bound_id=0
    )
    assert matrix.shape[0] == 1
    assert mat_bc.shape[0] == 1


def test_construct_boundary_value_matrices_shape_d():
    """shape_d argument enables coupling to a separate field shape."""
    n = 4
    x_f = np.linspace(0.0, 1.0, n + 1)
    shape = (n,)
    bc = {"a": 0, "b": 1, "d": 0.0}
    matrix, matrix_bc = construct_boundary_value_matrices(
        shape, x_f, bc=bc, bound_id=0, shape_d=(1,)
    )
    # shape_d=(1,) means the BC term is a single column
    assert matrix_bc.shape[1] == 1


def test_interp_cntr_to_stagg_tvd_no_limiter():
    """tvd_limiter=None should still return valid staggered values."""
    n = 10
    x_f = np.linspace(0, 1, n + 1)
    x_c = 0.5 * (x_f[:-1] + x_f[1:])
    c = np.arange(n, dtype=float)
    bc = ({"a": 0, "b": 1, "d": 0.0}, {"a": 0, "b": 1, "d": float(n - 1)})
    v = 1.0
    result, delta = interp_cntr_to_stagg_tvd(c, x_f, x_c, bc, v, tvd_limiter=None)
    assert result.shape[0] == n + 1
    # Without TVD: delta should be zero everywhere
    np.testing.assert_allclose(delta, 0.0, atol=1e-14)


def test_create_staggered_array_1d_to_staggered():
    """1-D array matching cell-center size should be interpolated to staggered."""
    n = 6
    shape = (n,)
    x_f = np.linspace(0, 1, n + 1)
    x_c = 0.5 * (x_f[:-1] + x_f[1:])
    arr = np.arange(n, dtype=float)
    result = create_staggered_array(arr, shape, 0, x_f=x_f, x_c=x_c)
    assert result.shape == (n + 1,)
