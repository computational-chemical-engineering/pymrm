"""
Operators Submodule for pymrm

This submodule provides numerical operators for spatial discretization,
including gradient and divergence operators.
These operators are essential for constructing finite difference
and finite volume schemes used in multiphase reactor modeling.

Functions:
- construct_grad: Constructs the gradient matrix for spatial differentiation.
- construct_grad_int: Constructs the gradient matrix for internal faces.
- construct_grad_bc: Constructs the gradient matrix for boundary faces.
- construct_div: Constructs the divergence matrix for flux calculations.

Dependencies:
- numpy
- scipy.sparse
- pymrm.grid (for optional grid generation)
- pymrm.helpers (for boundary condition handling)
"""

import math
import numpy as np
from scipy.sparse import csc_array, csr_array
from pymrm.helpers import unwrap_bc_coeff, _sparse_array
from pymrm.grid import generate_grid


def construct_grad(
    shape, x_f, x_c=None, bc=(None, None), axis=0, shapes_d=(None, None), format="csc"
):
    """
    Construct the gradient matrix for spatial differentiation.

    Parameters:
        shape (tuple or int): Shape of the domain. If an integer is provided, it is converted to a tuple.
        x_f (ndarray): Face positions.
        x_c (ndarray, optional): Cell center coordinates. If not provided, they are calculated.
        bc (tuple, optional): Boundary conditions as a tuple of dictionaries. Default is (None, None).
        axis (int, optional): Axis of differentiation. Default is 0.
        shapes_d (tuple, optional): Shapes for boundary condition contributions. Default is (None, None).
        format (str, optional): Sparse format, ``'csc'`` (default) or ``'csr'``.

    Returns:
        csc_array or csr_array: Gradient matrix.
        csc_array or csr_array: Gradient BC contribution (tuple if ``shapes_d`` is provided).
    """
    if isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)
    x_f, x_c = generate_grid(shape[axis], x_f, generate_x_c=True, x_c=x_c)
    grad_matrix = construct_grad_int(shape, x_f, x_c, axis, format=format)

    if bc == (None, None):
        shape_f = shape[:axis] + (shape[axis] + 1,) + shape[axis + 1:]
        grad_bc = csc_array((math.prod(shape_f), 1))
        return grad_matrix, grad_bc
    else:
        if shapes_d is None or shapes_d == (None, None):
            grad_matrix_bc, grad_bc = construct_grad_bc(
                shape, x_f, x_c, bc, axis, format=format
            )
            grad_matrix += grad_matrix_bc
            return grad_matrix, grad_bc
        else:
            grad_matrix_bc_0, grad_bc_0, grad_matrix_bc_1, grad_bc_1 = (
                construct_grad_bc(
                    shape, x_f, x_c, bc, axis, shapes_d=shapes_d, format=format
                )
            )
            grad_matrix += grad_matrix_bc_0 + grad_matrix_bc_1
            return grad_matrix, grad_bc_0, grad_bc_1


def construct_grad_int(shape, x_f, x_c=None, axis=0, format="csc"):
    """
    Construct the gradient matrix for internal faces.

    Parameters:
        shape (tuple): Shape of the domain.
        x_f (ndarray): Face coordinates.
        x_c (ndarray, optional): Cell center coordinates. If not provided, they are calculated.
        axis (int, optional): Axis of differentiation. Default is 0.
        format (str, optional): Sparse format, ``'csc'`` (default) or ``'csr'``.

    Returns:
        csc_array or csr_array: Gradient matrix for internal faces.
    """
    if axis < 0:
        axis += len(shape)
    shape_t = [
        math.prod(shape[:axis]),
        math.prod(shape[axis: axis + 1]),
        math.prod(shape[axis + 1:]),
    ]

    n0, n1, n2 = shape_t
    # Open grids for each dimension; the 4th axis selects the face-pair [j, j+1]
    i0 = np.arange(n0).reshape(-1, 1, 1, 1)
    i1 = np.arange(n1).reshape(1, -1, 1, 1)
    i2 = np.arange(n2).reshape(1, 1, -1, 1)
    face_pair = np.array([0, 1]).reshape(1, 1, 1, -1)
    i_f = np.ravel_multi_index((i0, i1 + face_pair, i2), (n0, n1 + 1, n2))

    if x_c is None:
        x_c = 0.5 * (x_f[:-1] + x_f[1:])

    dx_inv = np.tile(
        1 / (x_c[1:] - x_c[:-1]).reshape((1, -1, 1)), (n0, 1, n2)
    )
    values = np.empty(i_f.shape)
    values[:, 0, :, 0] = np.zeros((n0, n2))
    values[:, 1:, :, 0] = dx_inv
    values[:, :-1, :, 1] = -dx_inv
    values[:, -1, :, 1] = np.zeros((n0, n2))
    if format == "csc":
        grad_matrix = csc_array(
            (values.ravel(), i_f.ravel(), np.arange(0, i_f.size + 1, 2)),
            shape=(n0 * (n1 + 1) * n2, n0 * n1 * n2),
        )
    elif format == "csr":
        # Build CSR data in row (face) order with indptr.
        # Cell column indices: shape (n0, n1, n2)
        i_c = np.ravel_multi_index(
            (i0[..., 0], i1[..., 0], i2[..., 0]), (n0, n1, n2)
        )
        # Build (n0, n1+1, n2, 2) arrays: slot 0 = from cell j-1, slot 1 = from cell j
        csr_data = np.zeros((n0, n1 + 1, n2, 2))
        csr_cols = np.empty((n0, n1 + 1, n2, 2), dtype=np.intp)
        # Slot 0: right-side entry of cell j-1, contributes to face j (j=1..n1)
        csr_data[:, 1:, :, 0] = values[:, :, :, 1]
        csr_cols[:, 1:, :, 0] = i_c
        # Face 0 has no cell j-1; use cell 0's column so the dummy zero
        # entry lands on a valid column that already carries a zero value.
        csr_cols[:, 0, :, 0] = i_c[:, 0, :]
        # Slot 1: left-side entry of cell j, contributes to face j (j=0..n1-1)
        csr_data[:, :-1, :, 1] = values[:, :, :, 0]
        csr_cols[:, :-1, :, 1] = i_c
        # Face n1 has no cell j; use cell n1-1's column so the dummy zero
        # entry lands on a valid column that already carries a zero value.
        csr_cols[:, -1, :, 1] = i_c[:, -1, :]
        # Uniform 2 entries per row → simple indptr
        num_rows = n0 * (n1 + 1) * n2
        indptr = np.arange(0, 2 * num_rows + 1, 2, dtype=np.intp)
        grad_matrix = csr_array(
            (csr_data.ravel(), csr_cols.ravel(), indptr),
            shape=(n0 * (n1 + 1) * n2, n0 * n1 * n2),
        )
    else:
        raise ValueError(
            f"format must be 'csc' or 'csr', got {format!r}"
        )
    return grad_matrix


def construct_grad_bc(
    shape, x_f, x_c=None, bc=(None, None), axis=0, shapes_d=(None, None), format="csc"
):
    """
    Construct the gradient matrix for boundary faces.

    Parameters:
        shape (tuple): Shape of the domain.
        x_f (ndarray): Face coordinates.
        x_c (ndarray, optional): Cell center coordinates. If not provided, they are calculated.
        bc (tuple, optional): Boundary conditions as a tuple of dictionaries. Default is (None, None).
        axis (int, optional): Axis of differentiation. Default is 0.
        shapes_d (tuple, optional): Shapes for boundary condition contributions. Default is (None, None).
        format (str, optional): Sparse format, ``'csc'`` (default) or ``'csr'``.

    Returns:
        csc_array/csr_array or tuple: Gradient matrix for boundary faces and contributions from
            inhomogeneous boundary conditions.  If ``shapes_d`` is provided, returns a tuple of matrices.
    """
    shape_f = shape[:axis] + (shape[axis] + 1,) + shape[axis + 1:]
    shape_t = (math.prod(shape[:axis]), shape[axis], math.prod(shape[axis + 1:]))
    shape_f_t = (shape_t[0], shape_f[axis], shape_t[2])
    shape_bc = shape[:axis] + (1,) + shape[axis + 1:]
    shape_bc_d = (shape_t[0], shape_t[2])

    # Handle special case with one cell in the dimension axis.
    # This is convenient e.g. for flexibility where you can choose not to
    # spatially discretize a direction, but still use a BC, e.g. with a mass transfer coefficient
    # It is a bit subtle because in this case the two opposite faces influence each other
    n0, n1, n2 = shape_t
    i0 = np.arange(n0).reshape(-1, 1, 1)
    i2 = np.arange(n2).reshape(1, 1, -1)

    if n1 == 1:
        if x_c is None:
            x_c = 0.5 * (x_f[0:-1] + x_f[1:])
        # Both faces reference cell 0
        i_c = np.ravel_multi_index(
            (i0, np.array([0, 0]).reshape(1, -1, 1), i2), (n0, n1, n2)
        )
        i_f = np.ravel_multi_index(
            (i0, np.array([0, 1]).reshape(1, -1, 1), i2), shape_f_t
        )
        values = np.empty(shape_f_t)
        alpha_1 = (x_f[1] - x_f[0]) / ((x_c[0] - x_f[0]) * (x_f[1] - x_c[0]))
        alpha_2_left = (x_c[0] - x_f[0]) / ((x_f[1] - x_f[0]) * (x_f[1] - x_c[0]))
        alpha_0_left = alpha_1 - alpha_2_left
        alpha_2_right = -(x_c[0] - x_f[1]) / ((x_f[0] - x_f[1]) * (x_f[0] - x_c[0]))
        alpha_0_right = alpha_1 - alpha_2_right
        a, b, d = [
            [
                (
                    unwrap_bc_coeff(shape, bc_element[key], axis=axis)
                    if bc_element
                    else np.zeros((1,) * len(shape))
                )
                for bc_element in bc
            ]
            for key in ["a", "b", "d"]
        ]
        fctr = (b[0] + alpha_0_left * a[0]) * (
            b[1] + alpha_0_right * a[1]
        ) - alpha_2_left * alpha_2_right * a[0] * a[1]
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        value = np.broadcast_to(
            alpha_1 * b[0] * (a[1] * (alpha_0_right - alpha_2_left) + b[1]) * fctr,
            shape,
        )
        values[:, 0, :] = np.reshape(value, shape_bc_d)
        value = np.broadcast_to(
            alpha_1 * b[1] * (a[0] * (-alpha_0_left + alpha_2_right) - b[0]) * fctr,
            shape,
        )
        values[:, 1, :] = np.reshape(value, shape_bc_d)

        i_f_bc = np.ravel_multi_index(
            (i0, np.array([0, shape_f_t[1] - 1]).reshape(1, -1, 1), i2), shape_f_t
        )
        values_bc = np.empty((n0, 2, n2))
        value = np.broadcast_to(
            (
                (
                    a[1]
                    * (-alpha_0_left * alpha_0_right + alpha_2_left * alpha_2_right)
                    - alpha_0_left * b[1]
                )
                * d[0]
                - alpha_2_left * b[0] * d[1]
            )
            * fctr,
            shape_bc,
        )
        values_bc[:, 0, :] = np.reshape(value, shape_bc_d)
        value = np.broadcast_to(
            (
                (
                    a[0]
                    * (+alpha_0_left * alpha_0_right - alpha_2_left * alpha_2_right)
                    + alpha_0_right * b[0]
                )
                * d[1]
                + alpha_2_right * b[1] * d[0]
            )
            * fctr,
            shape_bc,
        )
        values_bc[:, 1, :] = np.reshape(value, shape_bc_d)
    else:
        # Cell indices: 2 near left boundary + 2 near right boundary
        cell_axis_idx = np.array([0, 1, n1 - 2, n1 - 1]).reshape(1, -1, 1)
        i_c = np.ravel_multi_index((i0, cell_axis_idx, i2), shape_t)
        # Face indices: left face (0) twice, right face (n1) twice
        nf = shape_f_t[1]
        face_axis_idx = np.array([0, 0, nf - 1, nf - 1]).reshape(1, -1, 1)
        i_f = np.ravel_multi_index((i0, face_axis_idx, i2), shape_f_t)
        i_f_bc = np.ravel_multi_index(
            (i0, np.array([0, nf - 1]).reshape(1, -1, 1), i2), shape_f_t
        )
        values_bc = np.empty((n0, 2, n2))
        values = np.empty((n0, 4, n2))
        if x_c is None:
            x_c = 0.5 * np.array(
                [x_f[0] + x_f[1], x_f[1] + x_f[2], x_f[-3] + x_f[-2], x_f[-2] + x_f[-1]]
            )

        # Get a, b, and d for left bc from dictionary
        alpha_1 = (x_c[1] - x_f[0]) / ((x_c[0] - x_f[0]) * (x_c[1] - x_c[0]))
        alpha_2 = (x_c[0] - x_f[0]) / ((x_c[1] - x_f[0]) * (x_c[1] - x_c[0]))
        alpha_0 = alpha_1 - alpha_2
        a, b, d = [
            (
                unwrap_bc_coeff(shape, bc[0][key], axis=axis)
                if bc[0]
                else np.zeros((1,) * len(shape))
            )
            for key in ["a", "b", "d"]
        ]
        b = b / alpha_0
        fctr = a + b
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        b_fctr = b * fctr
        b_fctr = np.broadcast_to(b_fctr, shape_bc).reshape(shape_bc_d)
        d_fctr = d * fctr
        d_fctr = np.broadcast_to(d_fctr, shape_bc).reshape(shape_bc_d)
        values[:, 0, :] = b_fctr * alpha_1
        values[:, 1, :] = -b_fctr * alpha_2
        values_bc[:, 0, :] = -d_fctr

        # Get a, b, and d for right bc from dictionary
        a, b, d = [
            (
                unwrap_bc_coeff(shape, bc[1][key], axis=axis)
                if bc[1]
                else np.zeros((1,) * len(shape))
            )
            for key in ["a", "b", "d"]
        ]
        alpha_1 = -(x_c[-2] - x_f[-1]) / ((x_c[-1] - x_f[-1]) * (x_c[-2] - x_c[-1]))
        alpha_2 = -(x_c[-1] - x_f[-1]) / ((x_c[-2] - x_f[-1]) * (x_c[-2] - x_c[-1]))
        alpha_0 = alpha_1 - alpha_2
        b = b / alpha_0
        fctr = a + b
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        b_fctr = b * fctr
        b_fctr = np.broadcast_to(b_fctr, shape_bc).reshape(shape_bc_d)
        d_fctr = d * fctr
        d_fctr = np.broadcast_to(d_fctr, shape_bc).reshape(shape_bc_d)
        values[:, -2, :] = b_fctr * alpha_2
        values[:, -1, :] = -b_fctr * alpha_1
        values_bc[:, -1, :] = d_fctr
    if (shapes_d[0] is None) and (shapes_d[1] is None):
        grad_bc = csc_array(
            (values_bc.ravel(), i_f_bc.ravel(), np.array([0, i_f_bc.size])),
            shape=(math.prod(shape_f_t), 1),
        )
        grad_matrix = _sparse_array(
            (values.ravel(), (i_f.ravel(), i_c.ravel())),
            shape=(math.prod(shape_f_t), math.prod(shape_t)),
            format=format,
        )
        return grad_matrix, grad_bc
    else:
        grad_bc = [None] * 2
        for i in range(2):
            if shapes_d[i] is None:
                shape_d = (1,) * len(shape_bc)
                num_cols = 1
            else:
                shape_d = shapes_d[i]
                num_cols = math.prod(shape_d)
            i_cols_bc = np.arange(num_cols, dtype=int).reshape(shape_d)
            i_cols_bc = np.broadcast_to(i_cols_bc, shape_bc)
            grad_bc[i] = csc_array(
                (
                    values_bc[:, i, :].ravel(),
                    (i_f_bc[:, i, :].ravel(), i_cols_bc.ravel()),
                ),
                shape=(math.prod(shape_f_t), num_cols),
            )
        if shape_t[1] == 1:
            grad_matrix_0 = _sparse_array(
                (values[:, 0, :].ravel(), (i_f[:, 0, :].ravel(), i_c[:, 0, :].ravel())),
                shape=(math.prod(shape_f_t), math.prod(shape_t)),
                format=format,
            )
            grad_matrix_1 = _sparse_array(
                (
                    values[:, -1, :].ravel(),
                    (i_f[:, -1, :].ravel(), i_c[:, -1, :].ravel()),
                ),
                shape=(math.prod(shape_f_t), math.prod(shape_t)),
                format=format,
            )
        else:
            grad_matrix_0 = _sparse_array(
                (
                    values[:, :2, :].ravel(),
                    (i_f[:, :2, :].ravel(), i_c[:, :2, :].ravel()),
                ),
                shape=(math.prod(shape_f_t), math.prod(shape_t)),
                format=format,
            )
            grad_matrix_1 = _sparse_array(
                (
                    values[:, -2:, :].ravel(),
                    (i_f[:, -2:, :].ravel(), i_c[:, -2:, :].ravel()),
                ),
                shape=(math.prod(shape_f_t), math.prod(shape_t)),
                format=format,
            )
        return grad_matrix_0, grad_bc[0], grad_matrix_1, grad_bc[1]


def construct_div(shape, x_f, nu=0, axis=0, format="csc"):
    """
    Construct the divergence matrix for flux calculations.

    Parameters:
        shape (tuple or int): Shape of the domain. If an integer is provided, it is converted to a tuple.
        x_f (ndarray): Face positions.
        nu (int or callable, optional): Geometry factor (0: flat, 1: cylindrical,
            2: spherical, or a callable for custom geometry). Default is 0.
        axis (int, optional): Axis along which divergence is computed. Default is 0.
        format (str, optional): Sparse format, ``'csc'`` (default) or ``'csr'``.

    Returns:
        csc_array or csr_array: Divergence matrix.
    """
    if isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)
    x_f = generate_grid(shape[axis], x_f)

    shape_f = shape[:axis] + (shape[axis] + 1,) + shape[axis + 1:]
    shape_t = (math.prod(shape[:axis]), shape[axis], math.prod(shape[axis + 1:]))
    shape_f_t = (shape_t[0], shape_f[axis], shape_t[2])

    n0, n1, n2 = shape_t
    # Each cell references its left face [j] and right face [j+1]
    i0 = np.arange(n0).reshape(-1, 1, 1, 1)
    i1 = np.arange(n1).reshape(1, -1, 1, 1)
    i2 = np.arange(n2).reshape(1, 1, -1, 1)
    face_pair = np.array([0, 1]).reshape(1, 1, 1, -1)
    i_f = np.ravel_multi_index((i0, i1 + face_pair, i2), shape_f_t)

    if callable(nu):
        area = nu(x_f).ravel()
        inv_sqrt3 = 1 / np.sqrt(3)
        x_f_r = x_f.ravel()
        dx_f = x_f_r[1:] - x_f_r[:-1]
        dvol_inv = 1 / (
            (
                nu(x_f_r[:-1] + (0.5 - 0.5 * inv_sqrt3) * dx_f)
                + nu(x_f_r[:-1] + (0.5 + 0.5 * inv_sqrt3) * dx_f)
            )
            * 0.5
            * dx_f
        )
    elif nu == 0:
        area = np.ones(shape_f_t[1])
        dvol_inv = 1 / (x_f[1:] - x_f[:-1])
    else:
        area = np.power(x_f.ravel(), nu)
        vol = area * x_f.ravel() / (nu + 1)
        dvol_inv = 1 / (vol[1:] - vol[:-1])

    values = np.empty((shape_t[1], 2))
    values[:, 0] = -area[:-1] * dvol_inv
    values[:, 1] = area[1:] * dvol_inv
    values_per_axis = values  # per-axis values (n1, 2) before tiling
    values = np.tile(values.reshape((1, -1, 1, 2)), (shape_t[0], 1, shape_t[2]))

    num_cells = np.prod(shape_t, dtype=int)
    num_face_flat = np.prod(shape_f_t, dtype=int)
    if format == "csc":
        # Build CSC data in column (face) order with indptr.
        # Cell indices: shape (n0, n1, n2)
        i_c = np.ravel_multi_index(
            (i0[..., 0], i1[..., 0], i2[..., 0]), (n0, n1, n2)
        )
        # (n0, n1+1, n2, 2): slot 0 = from cell j-1, slot 1 = from cell j
        csc_data = np.zeros((n0, n1 + 1, n2, 2))
        csc_rows = np.empty((n0, n1 + 1, n2, 2), dtype=np.intp)
        # Slot 0: entry from cell j-1 (right face = face j), valid j=1..n1
        csc_data[:, 1:, :, 0] = values_per_axis[:, 1].reshape(1, n1, 1)
        csc_rows[:, 1:, :, 0] = i_c
        # Face 0 has no cell j-1; use cell 0's row so the dummy zero
        # entry lands on a valid row that already carries a zero value.
        csc_rows[:, 0, :, 0] = i_c[:, 0, :]
        # Slot 1: entry from cell j (left face = face j), valid j=0..n1-1
        csc_data[:, :-1, :, 1] = values_per_axis[:, 0].reshape(1, n1, 1)
        csc_rows[:, :-1, :, 1] = i_c
        # Face n1 has no cell j; use cell n1-1's row so the dummy zero
        # entry lands on a valid row that already carries a zero value.
        csc_rows[:, -1, :, 1] = i_c[:, -1, :]
        # Uniform 2 entries per column → simple indptr
        indptr = np.arange(0, 2 * num_face_flat + 1, 2, dtype=np.intp)
        div_matrix = csc_array(
            (csc_data.ravel(), csc_rows.ravel(), indptr),
            shape=(num_cells, num_face_flat),
        )
    elif format == "csr":
        # Uniform 2 entries per row (each cell references left + right face)
        indptr = np.arange(0, 2 * num_cells + 1, 2, dtype=np.intp)
        div_matrix = csr_array(
            (values.ravel(), i_f.ravel(), indptr),
            shape=(num_cells, num_face_flat),
        )
    else:
        raise ValueError(
            f"format must be 'csc' or 'csr', got {format!r}"
        )
    div_matrix.sort_indices()
    return div_matrix
