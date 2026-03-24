import pytest
import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import spsolve
from pymrm.operators import (
    construct_grad,
    construct_grad_int,
    construct_grad_bc,
    construct_div,
)
from pymrm.grid import generate_grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_1d(n=10):
    shape = (n,)
    x_f = np.linspace(0, 1, n + 1)
    _, x_c = generate_grid(n, x_f, generate_x_c=True)
    return shape, x_f, x_c


@pytest.fixture(params=["csc", "csr"])
def sparse_format(request):
    return request.param


# ---------------------------------------------------------------------------
# Basic construction tests (existing)
# ---------------------------------------------------------------------------

def test_construct_grad_and_div(sparse_format):
    shape = (10,)
    x_f = np.linspace(0, 1, shape[0] + 1)
    grad, grad_bc = construct_grad(shape, x_f, format=sparse_format)
    div = construct_div(shape, x_f, format=sparse_format)
    assert grad.shape[0] > 0
    assert div.shape[0] > 0


def test_construct_grad_int(sparse_format):
    shape = (10,)
    x_f = np.linspace(0, 1, shape[0] + 1)
    grad_int = construct_grad_int(shape, x_f, format=sparse_format)
    assert grad_int.shape[0] > 0


def test_construct_grad_bc(sparse_format):
    shape = (10,)
    x_f = np.linspace(0, 1, shape[0] + 1)
    bc = ({"a": 0, "b": 1, "d": 1}, {"a": 1, "b": 0, "d": 0})
    grad_matrix, grad_bc = construct_grad_bc(shape, x_f, bc=bc, format=sparse_format)
    assert grad_matrix.shape[0] > 0
    assert grad_bc.shape[0] > 0


def test_construct_div(sparse_format):
    shape = (10,)
    x_f = np.linspace(0, 1, shape[0] + 1)
    div = construct_div(shape, x_f, format=sparse_format)
    assert div.shape[0] > 0


# ---------------------------------------------------------------------------
# Shape and type checks
# ---------------------------------------------------------------------------

def test_grad_and_div_shapes_1d(sparse_format):
    n = 8
    shape, x_f, _ = _make_1d(n)
    grad, grad_bc = construct_grad(shape, x_f, format=sparse_format)
    div = construct_div(shape, x_f, format=sparse_format)
    # grad: (n+1) x n, div: n x (n+1)
    assert grad.shape == (n + 1, n)
    assert div.shape == (n, n + 1)
    assert issparse(grad)
    assert issparse(div)


def test_grad_int_shape(sparse_format):
    n = 6
    shape, x_f, x_c = _make_1d(n)
    grad_int = construct_grad_int(shape, x_f, x_c, format=sparse_format)
    assert grad_int.shape == (n + 1, n)


def test_construct_grad_accepts_int_shape(sparse_format):
    """construct_grad should accept a plain int for shape."""
    x_f = np.linspace(0, 1, 6)
    grad, _ = construct_grad(5, x_f, format=sparse_format)
    assert grad.shape[1] == 5


# ---------------------------------------------------------------------------
# Accuracy on known problems
# ---------------------------------------------------------------------------

def test_grad_linear_profile_internal(sparse_format):
    """Gradient of a linear profile should equal the slope everywhere."""
    n = 10
    shape, x_f, x_c = _make_1d(n)
    slope = 3.0
    c = slope * x_c
    grad_int = construct_grad_int(shape, x_f, x_c, format=sparse_format)
    # Only internal faces (exclude boundary rows set to zero)
    face_grad = grad_int @ c
    # Internal faces 1..n-1 should have gradient ≈ slope
    np.testing.assert_allclose(face_grad[1:-1], slope, atol=1e-10)


def test_div_of_constant_flux_is_zero(sparse_format):
    """Divergence of a constant flux vector should be zero."""
    n = 10
    shape, x_f, _ = _make_1d(n)
    div = construct_div(shape, x_f, format=sparse_format)
    flux = np.ones(n + 1)
    result = div @ flux
    np.testing.assert_allclose(result, 0.0, atol=1e-12)


def test_laplacian_linear_profile(sparse_format):
    """Second-order Laplacian of a linear profile should be zero."""
    n = 12
    shape, x_f, x_c = _make_1d(n)
    bc = ({"a": 0, "b": 1, "d": 0.0}, {"a": 0, "b": 1, "d": 1.0})
    grad, grad_bc = construct_grad(shape, x_f, bc=bc, format=sparse_format)
    div = construct_div(shape, x_f, format=sparse_format)
    c = x_c  # exact linear solution
    # Assemble the residual: A*c + b_bc = 0
    A = div @ grad
    b_bc = (div @ grad_bc).toarray().ravel()
    residual = A @ c + b_bc
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Single-cell special case
# ---------------------------------------------------------------------------

def test_construct_grad_bc_single_cell(sparse_format):
    """Single-cell domain triggers the special-case branch in construct_grad_bc."""
    shape = (1,)
    x_f = np.array([0.0, 1.0])
    bc = ({"a": 0, "b": 1, "d": 0.0}, {"a": 0, "b": 1, "d": 1.0})
    grad_matrix, grad_bc = construct_grad_bc(shape, x_f, bc=bc, format=sparse_format)
    assert grad_matrix.shape[0] > 0
    assert grad_bc.shape[0] > 0


def test_construct_convflux_single_cell_operators(sparse_format):
    """Grad with single cell should return valid sparse matrices."""
    shape = (1,)
    x_f = np.array([0.0, 1.0])
    bc = ({"a": 0, "b": 1, "d": 2.0}, {"a": 0, "b": 1, "d": 5.0})
    grad, grad_bc = construct_grad(shape, x_f, bc=bc, format=sparse_format)
    assert issparse(grad)
    assert issparse(grad_bc)


# ---------------------------------------------------------------------------
# 2-D domain
# ---------------------------------------------------------------------------

def test_grad_div_2d_shape(sparse_format):
    """2D operators should produce matrices of correct shape."""
    shape = (5, 4)
    n_x, n_y = shape
    x_f = np.linspace(0, 1, n_x + 1)
    y_f = np.linspace(0, 2, n_y + 1)

    grad_x, _ = construct_grad(shape, x_f, axis=0, format=sparse_format)
    grad_y, _ = construct_grad(shape, y_f, axis=1, format=sparse_format)

    # Rows of grad_x: (n_x+1)*n_y, Cols: n_x*n_y
    assert grad_x.shape == ((n_x + 1) * n_y, n_x * n_y)
    assert grad_y.shape == (n_x * (n_y + 1), n_x * n_y)

    div_x = construct_div(shape, x_f, axis=0, format=sparse_format)
    div_y = construct_div(shape, y_f, axis=1, format=sparse_format)
    assert div_x.shape == (n_x * n_y, (n_x + 1) * n_y)
    assert div_y.shape == (n_x * n_y, n_x * (n_y + 1))


# ---------------------------------------------------------------------------
# Cylindrical and spherical geometry
# ---------------------------------------------------------------------------

def test_construct_div_cylindrical(sparse_format):
    """Cylindrical geometry (nu=1) should give valid divergence matrix."""
    n = 8
    shape = (n,)
    x_f = np.linspace(0.1, 1.0, n + 1)  # avoid r=0 singularity
    div = construct_div(shape, x_f, nu=1, format=sparse_format)
    assert issparse(div)
    assert div.shape == (n, n + 1)


def test_construct_div_spherical(sparse_format):
    """Spherical geometry (nu=2) should give valid divergence matrix."""
    n = 8
    shape = (n,)
    x_f = np.linspace(0.1, 1.0, n + 1)
    div = construct_div(shape, x_f, nu=2, format=sparse_format)
    assert issparse(div)
    assert div.shape == (n, n + 1)


def test_construct_div_callable_nu(sparse_format):
    """Custom callable geometry should work."""
    n = 8
    shape = (n,)
    x_f = np.linspace(0.0, 1.0, n + 1)
    div = construct_div(shape, x_f, nu=lambda r: np.ones_like(r), format=sparse_format)
    assert issparse(div)
    assert div.shape == (n, n + 1)


# ---------------------------------------------------------------------------
# shapes_d argument
# ---------------------------------------------------------------------------

def test_construct_grad_with_shapes_d(sparse_format):
    """shapes_d argument enables coupled BC decomposition."""
    n = 6
    shape = (n,)
    x_f = np.linspace(0, 1, n + 1)
    bc = ({"a": 0, "b": 1, "d": 0.0}, {"a": 0, "b": 1, "d": 1.0})
    result = construct_grad(shape, x_f, bc=bc, shapes_d=((1,), (1,)),
                            format=sparse_format)
    # Should return 3 values: grad_matrix, grad_bc_0, grad_bc_1
    assert len(result) == 3
    grad_matrix, grad_bc_0, grad_bc_1 = result
    assert issparse(grad_matrix)
    assert issparse(grad_bc_0)
    assert issparse(grad_bc_1)


# ---------------------------------------------------------------------------
# Negative axis support
# ---------------------------------------------------------------------------

def test_construct_grad_int_negative_axis(sparse_format):
    """Negative axis should work (axis += len(shape))."""
    shape = (6, 4)
    x_f = np.linspace(0, 1, 5)  # y axis: n_y=4, n_y+1=5 faces
    grad_int = construct_grad_int(shape, x_f, axis=-1, format=sparse_format)
    assert grad_int.shape[0] > 0
    assert grad_int.shape == (6 * 5, 6 * 4)


# ---------------------------------------------------------------------------
# construct_div with integer shape
# ---------------------------------------------------------------------------

def test_construct_div_integer_shape(sparse_format):
    """construct_div should accept a plain int for shape."""
    x_f = np.linspace(0, 1, 9)
    div = construct_div(8, x_f, format=sparse_format)
    assert div.shape == (8, 9)
    assert issparse(div)


# ---------------------------------------------------------------------------
# construct_grad_bc with shapes_d where one side is None
# ---------------------------------------------------------------------------

def test_construct_grad_bc_shapes_d_one_side_none(sparse_format):
    """shapes_d with one side None, one side specified."""
    n = 6
    shape = (n,)
    x_f = np.linspace(0, 1, n + 1)
    bc = ({"a": 0, "b": 1, "d": 0.0}, {"a": 0, "b": 1, "d": 1.0})
    # shapes_d=(None, (1,)): left side inherits default, right side is shape (1,)
    result = construct_grad(shape, x_f, bc=bc, shapes_d=(None, (1,)),
                            format=sparse_format)
    assert len(result) == 3
    _, grad_bc_0, grad_bc_1 = result
    assert issparse(grad_bc_0)
    assert issparse(grad_bc_1)
    # Left BC has 1 column (default), right BC has 1 column as specified
    assert grad_bc_0.shape[1] == 1
    assert grad_bc_1.shape[1] == 1


def test_construct_grad_bc_shapes_d_single_cell(sparse_format):
    """shapes_d with single-cell domain."""
    shape = (1,)
    x_f = np.array([0.0, 1.0])
    bc = ({"a": 0, "b": 1, "d": 0.0}, {"a": 0, "b": 1, "d": 1.0})
    result = construct_grad(shape, x_f, bc=bc, shapes_d=((1,), (1,)),
                            format=sparse_format)
    assert len(result) == 3
