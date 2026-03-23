"""Tests verifying that all sparse-matrix-creating functions produce
numerically identical results for format='csr' and format='csc'.

Sparse vectors (Nx1 matrices) are always CSC regardless of format.
"""

import warnings
import numpy as np
import pytest
from scipy.sparse import issparse, csc_array, csr_array
from scipy.sparse.linalg import spsolve

from pymrm import (
    construct_grad,
    construct_grad_int,
    construct_grad_bc,
    construct_div,
    construct_convflux_upwind,
    construct_convflux_upwind_int,
    construct_convflux_bc,
    construct_boundary_value_matrices,
    construct_coefficient_matrix,
    construct_interface_matrices,
    translate_indices_to_larger_array,
    update_csc_array_indices,
    update_csr_array_indices,
    update_array_indices,
    NumJac,
)
from pymrm.grid import generate_grid


# ---------------------------------------------------------------------------
# helpers / utilities
# ---------------------------------------------------------------------------

def _assert_same_dense(mat_csc, mat_csr, atol=1e-14):
    """Assert that a CSC and a CSR matrix have the same dense representation."""
    np.testing.assert_allclose(
        mat_csr.toarray(), mat_csc.toarray(), atol=atol,
        err_msg="CSR and CSC dense representations differ"
    )


def _make_1d(n=10):
    shape = (n,)
    x_f = np.linspace(0, 1, n + 1)
    _, x_c = generate_grid(n, x_f, generate_x_c=True)
    return shape, x_f, x_c


# ---------------------------------------------------------------------------
# construct_grad / construct_grad_int / construct_grad_bc
# ---------------------------------------------------------------------------

class TestGradCSR:
    def test_grad_int_csr_matches_csc(self):
        shape, x_f, x_c = _make_1d(10)
        g_csc = construct_grad_int(shape, x_f, x_c, format="csc")
        g_csr = construct_grad_int(shape, x_f, x_c, format="csr")
        assert isinstance(g_csr, csr_array)
        assert isinstance(g_csc, csc_array)
        _assert_same_dense(g_csc, g_csr)

    def test_grad_bc_csr_matches_csc(self):
        shape, x_f, x_c = _make_1d(8)
        bc = ({"a": 0, "b": 1, "d": 1}, {"a": 1, "b": 0, "d": 0})
        m_csc, b_csc = construct_grad_bc(shape, x_f, x_c, bc=bc, format="csc")
        m_csr, b_csr = construct_grad_bc(shape, x_f, x_c, bc=bc, format="csr")
        assert isinstance(m_csr, csr_array)
        # BC vectors are always CSC
        assert isinstance(b_csc, csc_array)
        assert isinstance(b_csr, csc_array)
        _assert_same_dense(m_csc, m_csr)
        _assert_same_dense(b_csc, b_csr)

    def test_grad_full_csr_matches_csc(self):
        shape, x_f, _ = _make_1d(12)
        bc = ({"a": 0, "b": 1, "d": 0.0}, {"a": 0, "b": 1, "d": 1.0})
        g_csc, gb_csc = construct_grad(shape, x_f, bc=bc, format="csc")
        g_csr, gb_csr = construct_grad(shape, x_f, bc=bc, format="csr")
        assert isinstance(g_csr, csr_array)
        _assert_same_dense(g_csc, g_csr)
        _assert_same_dense(gb_csc, gb_csr)

    def test_grad_no_bc_csr(self):
        shape, x_f, _ = _make_1d(6)
        g_csc, gb_csc = construct_grad(shape, x_f, format="csc")
        g_csr, gb_csr = construct_grad(shape, x_f, format="csr")
        assert isinstance(g_csr, csr_array)
        _assert_same_dense(g_csc, g_csr)

    def test_grad_2d_csr(self):
        shape = (5, 4)
        x_f = np.linspace(0, 1, 6)
        g_csc, _ = construct_grad(shape, x_f, axis=0, format="csc")
        g_csr, _ = construct_grad(shape, x_f, axis=0, format="csr")
        _assert_same_dense(g_csc, g_csr)

    def test_grad_shapes_d_csr(self):
        shape, x_f, _ = _make_1d(6)
        bc = ({"a": 0, "b": 1, "d": 0.0}, {"a": 0, "b": 1, "d": 1.0})
        result_csc = construct_grad(shape, x_f, bc=bc, shapes_d=((1,), (1,)), format="csc")
        result_csr = construct_grad(shape, x_f, bc=bc, shapes_d=((1,), (1,)), format="csr")
        assert len(result_csr) == 3
        _assert_same_dense(result_csc[0], result_csr[0])

    def test_grad_bc_single_cell_csr(self):
        shape = (1,)
        x_f = np.array([0.0, 1.0])
        bc = ({"a": 0, "b": 1, "d": 0.0}, {"a": 0, "b": 1, "d": 1.0})
        m_csc, b_csc = construct_grad_bc(shape, x_f, bc=bc, format="csc")
        m_csr, b_csr = construct_grad_bc(shape, x_f, bc=bc, format="csr")
        _assert_same_dense(m_csc, m_csr)
        _assert_same_dense(b_csc, b_csr)


# ---------------------------------------------------------------------------
# construct_div
# ---------------------------------------------------------------------------

class TestDivCSR:
    def test_div_csr_matches_csc(self):
        shape, x_f, _ = _make_1d(10)
        d_csc = construct_div(shape, x_f, format="csc")
        d_csr = construct_div(shape, x_f, format="csr")
        assert isinstance(d_csr, csr_array)
        _assert_same_dense(d_csc, d_csr)

    def test_div_cylindrical_csr(self):
        shape = (8,)
        x_f = np.linspace(0.1, 1.0, 9)
        d_csc = construct_div(shape, x_f, nu=1, format="csc")
        d_csr = construct_div(shape, x_f, nu=1, format="csr")
        _assert_same_dense(d_csc, d_csr)

    def test_div_spherical_csr(self):
        shape = (8,)
        x_f = np.linspace(0.1, 1.0, 9)
        d_csc = construct_div(shape, x_f, nu=2, format="csc")
        d_csr = construct_div(shape, x_f, nu=2, format="csr")
        _assert_same_dense(d_csc, d_csr)

    def test_div_2d_csr(self):
        shape = (5, 4)
        x_f = np.linspace(0, 1, 6)
        d_csc = construct_div(shape, x_f, axis=0, format="csc")
        d_csr = construct_div(shape, x_f, axis=0, format="csr")
        _assert_same_dense(d_csc, d_csr)


# ---------------------------------------------------------------------------
# Laplacian accuracy (CSR path)
# ---------------------------------------------------------------------------

class TestLaplacianCSR:
    def test_laplacian_linear_csr(self):
        n = 12
        shape, x_f, x_c = _make_1d(n)
        bc = ({"a": 0, "b": 1, "d": 0.0}, {"a": 0, "b": 1, "d": 1.0})
        grad, grad_bc = construct_grad(shape, x_f, bc=bc, format="csr")
        div = construct_div(shape, x_f, format="csr")
        c = x_c
        A = div @ grad
        b_bc = (div @ grad_bc).toarray().ravel()
        residual = A @ c + b_bc
        np.testing.assert_allclose(residual, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# construct_convflux_upwind / construct_convflux_upwind_int / construct_convflux_bc
# ---------------------------------------------------------------------------

class TestConvectCSR:
    def test_convflux_upwind_int_csr(self):
        shape, x_f, x_c = _make_1d(10)
        c_csc = construct_convflux_upwind_int(shape, format="csc")
        c_csr = construct_convflux_upwind_int(shape, format="csr")
        assert isinstance(c_csr, csr_array)
        _assert_same_dense(c_csc, c_csr)

    def test_convflux_upwind_csr(self):
        shape, x_f, _ = _make_1d(10)
        bc = ({"a": 1, "b": 0, "d": 1}, {"a": 1, "b": 0, "d": 0})
        m_csc, b_csc = construct_convflux_upwind(shape, x_f, bc=bc, format="csc")
        m_csr, b_csr = construct_convflux_upwind(shape, x_f, bc=bc, format="csr")
        assert isinstance(m_csr, csr_array)
        _assert_same_dense(m_csc, m_csr)
        _assert_same_dense(b_csc, b_csr)

    def test_convflux_upwind_no_bc_csr(self):
        shape, x_f, _ = _make_1d(8)
        m_csc, b_csc = construct_convflux_upwind(shape, x_f, format="csc")
        m_csr, b_csr = construct_convflux_upwind(shape, x_f, format="csr")
        _assert_same_dense(m_csc, m_csr)

    def test_convflux_bc_csr(self):
        shape, x_f, x_c = _make_1d(8)
        bc = ({"a": 1, "b": 0, "d": 1}, {"a": 1, "b": 0, "d": 0})
        m_csc, b_csc = construct_convflux_bc(shape, x_f, x_c, bc=bc, format="csc")
        m_csr, b_csr = construct_convflux_bc(shape, x_f, x_c, bc=bc, format="csr")
        _assert_same_dense(m_csc, m_csr)
        _assert_same_dense(b_csc, b_csr)

    def test_convflux_bc_single_cell_csr(self):
        shape = (1,)
        x_f = np.array([0.0, 1.0])
        bc = ({"a": 0, "b": 1, "d": 2.0}, {"a": 0, "b": 1, "d": 5.0})
        m_csc, b_csc = construct_convflux_upwind(shape, x_f, bc=bc, format="csc")
        m_csr, b_csr = construct_convflux_upwind(shape, x_f, bc=bc, format="csr")
        _assert_same_dense(m_csc, m_csr)
        _assert_same_dense(b_csc, b_csr)

    def test_convflux_shapes_d_csr(self):
        shape, x_f, _ = _make_1d(6)
        bc = ({"a": 1, "b": 0, "d": 1}, {"a": 1, "b": 0, "d": 0})
        res_csc = construct_convflux_upwind(shape, x_f, bc=bc, shapes_d=((1,), (1,)), format="csc")
        res_csr = construct_convflux_upwind(shape, x_f, bc=bc, shapes_d=((1,), (1,)), format="csr")
        assert len(res_csr) == 3
        _assert_same_dense(res_csc[0], res_csr[0])


# ---------------------------------------------------------------------------
# construct_boundary_value_matrices
# ---------------------------------------------------------------------------

class TestBoundaryValueCSR:
    def test_boundary_value_matrices_csr(self):
        shape = (10,)
        x_f = np.linspace(0, 1, 11)
        bc = {"a": 1, "b": 0, "d": 1.0}
        m_csc, b_csc = construct_boundary_value_matrices(shape, x_f, bc=bc, format="csc")
        m_csr, b_csr = construct_boundary_value_matrices(shape, x_f, bc=bc, format="csr")
        assert isinstance(m_csr, csr_array)
        _assert_same_dense(m_csc, m_csr)
        _assert_same_dense(b_csc, b_csr)

    def test_boundary_value_matrices_bound_id_1_csr(self):
        shape = (8,)
        x_f = np.linspace(0, 1, 9)
        bc = {"a": 0, "b": 1, "d": 5.0}
        m_csc, b_csc = construct_boundary_value_matrices(shape, x_f, bc=bc, bound_id=1, format="csc")
        m_csr, b_csr = construct_boundary_value_matrices(shape, x_f, bc=bc, bound_id=1, format="csr")
        _assert_same_dense(m_csc, m_csr)
        _assert_same_dense(b_csc, b_csr)


# ---------------------------------------------------------------------------
# construct_coefficient_matrix
# ---------------------------------------------------------------------------

class TestCoefficientMatrixCSR:
    def test_flat_csr(self):
        coeffs = np.arange(5.0)
        m_csc = construct_coefficient_matrix(coeffs, format="csc")
        m_csr = construct_coefficient_matrix(coeffs, format="csr")
        assert isinstance(m_csr, csr_array)
        _assert_same_dense(m_csc, m_csr)

    def test_shape_csr(self):
        shape = (5, 4)
        coeffs = np.ones(shape) * 3.0
        m_csc = construct_coefficient_matrix(coeffs, shape=shape, format="csc")
        m_csr = construct_coefficient_matrix(coeffs, shape=shape, format="csr")
        _assert_same_dense(m_csc, m_csr)

    def test_pair_of_shapes_csr(self):
        shape_rows = (5, 1)
        shape_cols = (1, 4)
        coeffs = np.ones((5, 4))
        m_csc = construct_coefficient_matrix(coeffs, shape=(shape_rows, shape_cols), format="csc")
        m_csr = construct_coefficient_matrix(coeffs, shape=(shape_rows, shape_cols), format="csr")
        _assert_same_dense(m_csc, m_csr)


# ---------------------------------------------------------------------------
# coupling: update_array_indices / update_csr_array_indices / deprecation
# ---------------------------------------------------------------------------

class TestCouplingCSR:
    def test_update_csc_array_indices_deprecation(self):
        """update_csc_array_indices should emit a DeprecationWarning."""
        from scipy.sparse import csc_matrix
        mat = csc_matrix(np.eye(3))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = update_csc_array_indices(mat, (3,), (6,))
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
        assert result.shape == (6, 6)

    def test_update_csr_array_indices_basic(self):
        mat = csr_array(np.eye(3))
        result = update_csr_array_indices(mat, (3,), (6,))
        assert isinstance(result, csr_array)
        assert result.shape == (6, 6)
        assert abs(result.sum() - 3.0) < 1e-12

    def test_update_csr_array_indices_with_offset(self):
        mat = csr_array(np.array([[1.0, 0.0], [0.0, 2.0]]))
        result = update_csr_array_indices(mat, (2,), (5,), offset=(2,))
        assert result.shape == (5, 5)
        assert abs(result.sum() - 3.0) < 1e-12

    def test_update_csr_matches_csc(self):
        """CSR and CSC update should yield the same dense matrix."""
        dense = np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0], [5.0, 0.0, 6.0]])
        mat_csc = csc_array(dense)
        mat_csr = csr_array(dense)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            res_csc = update_csc_array_indices(mat_csc, (3,), (5,))
        res_csr = update_csr_array_indices(mat_csr, (3,), (5,))
        _assert_same_dense(res_csc, res_csr)

    def test_update_csr_with_offset_matches_csc(self):
        dense = np.diag([1.0, 2.0, 3.0])
        mat_csc = csc_array(dense)
        mat_csr = csr_array(dense)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            res_csc = update_csc_array_indices(mat_csc, (3,), (6,), offset=(2,))
        res_csr = update_csr_array_indices(mat_csr, (3,), (6,), offset=(2,))
        _assert_same_dense(res_csc, res_csr)

    def test_update_array_indices_dispatches_csc(self):
        mat = csc_array(np.eye(3))
        result = update_array_indices(mat, (3,), (5,))
        assert isinstance(result, csc_array)
        assert result.shape == (5, 5)

    def test_update_array_indices_dispatches_csr(self):
        mat = csr_array(np.eye(3))
        result = update_array_indices(mat, (3,), (5,))
        assert isinstance(result, csr_array)
        assert result.shape == (5, 5)


# ---------------------------------------------------------------------------
# construct_interface_matrices
# ---------------------------------------------------------------------------

class TestInterfaceCSR:
    def test_interface_matrices_csr(self):
        n = 5
        shapes = ((n,), (n,))
        x_f1 = np.linspace(0, 1, n + 1)
        x_f2 = np.linspace(1, 2, n + 1)
        res_csc = construct_interface_matrices(shapes, (x_f1, x_f2), format="csc")
        res_csr = construct_interface_matrices(shapes, (x_f1, x_f2), format="csr")
        assert len(res_csr) == 4
        # Interface matrices are in requested format
        assert isinstance(res_csr[0], csr_array)
        assert isinstance(res_csr[2], csr_array)
        # BC vectors stay as csc
        assert isinstance(res_csr[1], csc_array)
        assert isinstance(res_csr[3], csc_array)
        # Numerically identical
        _assert_same_dense(res_csc[0], res_csr[0])
        _assert_same_dense(res_csc[1], res_csr[1])
        _assert_same_dense(res_csc[2], res_csr[2])
        _assert_same_dense(res_csc[3], res_csr[3])

    def test_interface_flux_continuity_csr(self):
        n = 8
        shapes = ((n,), (n,))
        x_f1 = np.linspace(0, 1, n + 1)
        x_f2 = np.linspace(1, 2, n + 1)
        im0, _, im1, _ = construct_interface_matrices(shapes, (x_f1, x_f2), format="csr")
        c_left = np.linspace(0.5, 1.0, n)
        c_right = np.linspace(1.0, 1.5, n)
        c = np.concatenate([c_left, c_right])
        val_left = np.asarray(im0 @ c).ravel()
        val_right = np.asarray(im1 @ c).ravel()
        np.testing.assert_allclose(val_left, val_right, atol=1e-10)


# ---------------------------------------------------------------------------
# NumJac
# ---------------------------------------------------------------------------

class TestNumJacCSR:
    def test_numjac_csr(self):
        shape = (5,)
        nj_csc = NumJac(shape=shape, format="csc")
        nj_csr = NumJac(shape=shape, format="csr")
        f = lambda c: c ** 2
        c = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, jac_csc = nj_csc(f, c)
        _, jac_csr = nj_csr(f, c)
        assert isinstance(jac_csr, csr_array)
        _assert_same_dense(jac_csc, jac_csr, atol=1e-8)


# ---------------------------------------------------------------------------
# Invalid format raises ValueError
# ---------------------------------------------------------------------------

class TestInvalidFormat:
    def test_invalid_format_grad(self):
        shape, x_f, _ = _make_1d(5)
        with pytest.raises(ValueError, match="format"):
            construct_grad(shape, x_f, format="coo")

    def test_invalid_format_div(self):
        shape, x_f, _ = _make_1d(5)
        with pytest.raises(ValueError, match="format"):
            construct_div(shape, x_f, format="bad")

    def test_invalid_format_coefficient(self):
        with pytest.raises(ValueError, match="format"):
            construct_coefficient_matrix(np.ones(5), format="xyz")
