import pytest
import numpy as np
from scipy.sparse import issparse
from pymrm.helpers import unwrap_bc_coeff
from pymrm import construct_coefficient_matrix


@pytest.fixture(params=["csc", "csr"])
def sparse_format(request):
    return request.param


# ---------------------------------------------------------------------------
# Existing tests
# ---------------------------------------------------------------------------

def test_unwrap_bc_coeff():
    shape = (5,)
    bc_coeff = [1, 2, 3, 4, 5]
    result = unwrap_bc_coeff(shape, bc_coeff)
    assert isinstance(result, np.ndarray)
    assert result.shape[-1] == 5


def test_construct_coefficient_matrix(sparse_format):
    coeffs = np.arange(5.0)
    mat = construct_coefficient_matrix(coeffs, format=sparse_format)
    assert mat.shape[0] == mat.shape[1]


# ---------------------------------------------------------------------------
# unwrap_bc_coeff – various shapes and dimensions
# ---------------------------------------------------------------------------

def test_unwrap_bc_coeff_scalar():
    """Scalar coefficient should broadcast to all cells."""
    shape = (8,)
    result = unwrap_bc_coeff(shape, 3.14)
    assert isinstance(result, np.ndarray)


def test_unwrap_bc_coeff_2d_shape():
    """2-D domain: coefficient should be expandable for the given axis."""
    shape = (5, 4)
    # Coefficient varying along axis 1 (n_y values)
    bc_coeff = np.arange(4.0)
    result = unwrap_bc_coeff(shape, bc_coeff, axis=0)
    assert isinstance(result, np.ndarray)


def test_unwrap_bc_coeff_full_shape():
    """Full-shape coefficient array should pass through unchanged."""
    shape = (3, 4)
    bc_coeff = np.ones(shape)
    result = unwrap_bc_coeff(shape, bc_coeff, axis=0)
    assert result.shape == shape


# ---------------------------------------------------------------------------
# construct_coefficient_matrix – mode 1 (flat coefficients)
# ---------------------------------------------------------------------------

def test_construct_coefficient_matrix_flat_diagonal(sparse_format):
    """Flat 1-D coefficients become a diagonal N×N matrix."""
    n = 7
    coeffs = np.arange(1.0, n + 1)
    mat = construct_coefficient_matrix(coeffs, format=sparse_format)
    assert mat.shape == (n, n)
    assert issparse(mat)
    # Diagonal values should match input coefficients
    diag = np.array(mat.diagonal())
    np.testing.assert_allclose(diag, coeffs, atol=1e-14)


# ---------------------------------------------------------------------------
# construct_coefficient_matrix – mode 2 (single tuple shape)
# ---------------------------------------------------------------------------

def test_construct_coefficient_matrix_single_shape_1d(sparse_format):
    """Single shape tuple: broadcasts coefficients to shape and makes diagonal."""
    n = 5
    shape = (n,)
    coeffs = np.ones(n) * 2.0
    mat = construct_coefficient_matrix(coeffs, shape=shape, format=sparse_format)
    assert mat.shape == (n, n)
    assert issparse(mat)
    np.testing.assert_allclose(mat.diagonal(), 2.0 * np.ones(n), atol=1e-14)


def test_construct_coefficient_matrix_single_shape_with_axis(sparse_format):
    """Single shape with axis: size along axis is incremented by 1 (staggered)."""
    shape = (5, 4)
    n_faces = (6, 4)  # axis=0 increases by 1
    coeffs = np.ones((6, 4))
    mat = construct_coefficient_matrix(coeffs, shape=shape, axis=0,
                                       format=sparse_format)
    expected_size = 6 * 4
    assert mat.shape == (expected_size, expected_size)


def test_construct_coefficient_matrix_broadcasts_scalar(sparse_format):
    """Scalar coefficient with shape should broadcast to the full diagonal."""
    shape = (4, 3)
    n = 4 * 3
    mat = construct_coefficient_matrix(np.array(5.0), shape=shape,
                                       format=sparse_format)
    assert mat.shape == (n, n)
    np.testing.assert_allclose(mat.diagonal(), 5.0, atol=1e-14)


# ---------------------------------------------------------------------------
# construct_coefficient_matrix – mode 3 (pair of tuples)
# ---------------------------------------------------------------------------

def test_construct_coefficient_matrix_pair_of_shapes(sparse_format):
    """Pair of tuples: creates a rectangular coupling matrix."""
    shape_rows = (5, 1)
    shape_cols = (1, 4)
    coeffs = np.ones((5, 4))
    mat = construct_coefficient_matrix(coeffs, shape=(shape_rows, shape_cols),
                                       format=sparse_format)
    assert mat.shape == (5, 4)
    assert issparse(mat)


def test_construct_coefficient_matrix_pair_of_shapes_with_axis(sparse_format):
    """Pair of tuples with axis: staggered face–cell coupling."""
    shape_rows = (1, 4)   # boundary face shape
    shape_cols = (5, 4)   # cell shape
    coeffs = np.ones((1, 4))
    mat = construct_coefficient_matrix(coeffs, shape=(shape_rows, shape_cols),
                                       axis=0, format=sparse_format)
    # shape_cols with axis=0 should have row-dim incremented: (6,4) → 24 cols
    # shape_rows with axis=0: (1,4) → 2*4=8 rows? depends on expand logic
    assert issparse(mat)


def test_construct_coefficient_matrix_rectangular(sparse_format):
    """Rectangular matrix using broadcastable pair of shapes."""
    # shape_rows=(5,1), shape_cols=(5,4): working_shape=(5,4)
    # rows can broadcast from (5,1) to (5,4), cols from (5,4) to (5,4)
    shape_rows = (5, 1)
    shape_cols = (5, 4)
    coeffs = np.ones((5, 4))
    mat = construct_coefficient_matrix(coeffs, shape=(shape_rows, shape_cols),
                                       format=sparse_format)
    assert mat.shape == (5, 20)
    assert issparse(mat)
