import pytest
import numpy as np
from scipy.sparse import csc_matrix, issparse
from pymrm import (
    translate_indices_to_larger_array,
    update_csc_array_indices,
    construct_interface_matrices,
)


# ---------------------------------------------------------------------------
# Existing tests
# ---------------------------------------------------------------------------

def test_translate_indices_to_larger_array():
    indices = np.array([0, 1, 2])
    shape = (3,)
    new_shape = (6,)
    result = translate_indices_to_larger_array(indices, shape, new_shape)
    assert np.all(result >= 0)


def test_update_csc_array_indices():
    mat = csc_matrix(np.eye(3))
    shape = (3,)
    new_shape = (6,)
    result = update_csc_array_indices(mat, shape, new_shape)
    assert result.shape[0] == 6
    assert result.shape[1] == 6


# ---------------------------------------------------------------------------
# translate_indices_to_larger_array
# ---------------------------------------------------------------------------

def test_translate_indices_no_offset():
    """Without offset, indices should remain the same (0-based embedding)."""
    indices = np.array([0, 1, 2])
    shape = (3,)
    new_shape = (10,)
    result = translate_indices_to_larger_array(indices, shape, new_shape)
    np.testing.assert_array_equal(result, indices)


def test_translate_indices_with_offset():
    """With an offset, indices should be shifted."""
    indices = np.array([0, 1, 2])
    shape = (3,)
    new_shape = (10,)
    offset = (3,)
    result = translate_indices_to_larger_array(indices, shape, new_shape, offset)
    np.testing.assert_array_equal(result, indices + 3)


def test_translate_indices_2d():
    """Multi-dimensional index translation should work correctly."""
    shape = (2, 3)
    new_shape = (4, 6)
    linear_indices = np.array([0, 1, 2, 3, 4, 5])
    offset = (1, 2)
    result = translate_indices_to_larger_array(linear_indices, shape, new_shape, offset)
    # All translated indices should be within [0, 24)
    assert np.all(result >= 0)
    assert np.all(result < np.prod(new_shape))


# ---------------------------------------------------------------------------
# update_csc_array_indices
# ---------------------------------------------------------------------------

def test_update_csc_array_indices_values_preserved():
    """Data values should not change after index translation."""
    mat = csc_matrix(np.diag([1.0, 2.0, 3.0]))
    result = update_csc_array_indices(mat, (3,), (5,))
    # The original values should be in the result
    assert abs(result.sum() - mat.sum()) < 1e-12


def test_update_csc_array_indices_with_offset():
    """With offset, values should appear at shifted positions."""
    mat = csc_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
    shape = (2,)
    new_shape = (5,)
    offset = (2,)
    result = update_csc_array_indices(mat, shape, new_shape, offset)
    assert result.shape == (5, 5)
    assert abs(result.sum() - mat.sum()) < 1e-12


# ---------------------------------------------------------------------------
# construct_interface_matrices
# ---------------------------------------------------------------------------

def test_construct_interface_matrices_basic():
    """Basic interface matrix construction should return 4 sparse arrays."""
    n1, n2 = 5, 5
    shapes = ((n1,), (n2,))
    x_f1 = np.linspace(0, 1, n1 + 1)
    x_f2 = np.linspace(1, 2, n2 + 1)
    x_fs = (x_f1, x_f2)

    result = construct_interface_matrices(shapes, x_fs)
    # Should return (interface_matrix_0, interface_bc_0, interface_matrix_1, interface_bc_1)
    assert len(result) == 4
    for mat in result:
        assert issparse(mat)


def test_construct_interface_matrices_shapes():
    """Interface matrices should have the correct shapes."""
    n1, n2 = 4, 6
    shapes = ((n1,), (n2,))
    x_f1 = np.linspace(0, 1, n1 + 1)
    x_f2 = np.linspace(1, 3, n2 + 1)
    x_fs = (x_f1, x_f2)

    im0, ibc0, im1, ibc1 = construct_interface_matrices(shapes, x_fs)

    # Interface matrix maps from (n1+n2,) to interface (1 point per interface)
    assert im0.shape[1] == n1 + n2
    assert im1.shape[1] == n1 + n2


def test_construct_interface_matrices_flux_continuity():
    """Default IC ensures flux continuity at the interface."""
    n = 8
    shapes = ((n,), (n,))
    x_f1 = np.linspace(0, 1, n + 1)
    x_f2 = np.linspace(1, 2, n + 1)
    x_fs = (x_f1, x_f2)

    im0, ibc0, im1, ibc1 = construct_interface_matrices(shapes, x_fs)

    # With a linear concentration profile across the interface,
    # both interface matrices should give the same interface value
    c_left = np.linspace(0.5, 1.0, n)
    c_right = np.linspace(1.0, 1.5, n)
    c = np.concatenate([c_left, c_right])

    val_left = np.asarray(im0 @ c).ravel()
    val_right = np.asarray(im1 @ c).ravel()

    # Both sides should agree on the interface value (flux continuity)
    np.testing.assert_allclose(val_left, val_right, atol=1e-10)


def test_construct_interface_matrices_mismatched_shapes_raises():
    """Shapes that differ on non-interface axes should raise ValueError."""
    shapes = ((5, 3), (5, 4))  # different y-sizes
    x_f1 = np.linspace(0, 1, 6)
    x_f2 = np.linspace(1, 2, 6)
    with pytest.raises(ValueError):
        construct_interface_matrices(shapes, (x_f1, x_f2))


def test_construct_interface_matrices_with_shapes_d():
    """shapes_d enables extended return signature for coupled BC decomposition."""
    n = 4
    shapes = ((n,), (n,))
    x_f1 = np.linspace(0, 1, n + 1)
    x_f2 = np.linspace(1, 2, n + 1)
    x_fs = (x_f1, x_f2)

    result = construct_interface_matrices(
        shapes, x_fs, shapes_d=((1,), (1,))
    )
    # With shapes_d, returns 6 elements
    assert len(result) == 6
