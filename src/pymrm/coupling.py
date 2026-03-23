"""
Coupling Subpackage for pymrm
=============================

This subpackage provides utilities for constructing and managing the coupling of multiple subdomains
in a computational model. It supports translating indices between different domains, updating sparse
matrices to accommodate larger domains, and constructing interface matrices for implicit coupling.

Key functionalities:
- **translate_indices_to_larger_array**: Translates indices from a subdomain to a larger array.
- **update_csc_array_indices**: Adjusts sparse matrix indices to fit a larger domain.
- **construct_interface_matrices**: Constructs sparse matrices for interface conditions between coupled domains.

The provided functions are particularly useful for implicit coupling approaches in computational fluid
dynamics and other numerical simulations requiring domain decomposition.
"""

import math
import numbers
import numpy as np
from scipy.sparse import csc_array
from pymrm.helpers import unwrap_bc_coeff


def translate_indices_to_larger_array(linear_indices, shape, new_shape, offset=None):
    """
    Translate linear indices from a subarray to their corresponding indices in a larger array.

    Parameters:
    -----------
    linear_indices : array-like
        Linear indices in the subarray.
    shape : tuple
        Shape of the subarray.
    new_shape : tuple
        Shape of the larger ND array.
    offset : tuple, optional
        Offset of the subarray’s top-left corner in the larger array (default: None).

    Returns:
    --------
    np.ndarray
        Linear indices in the larger ND array.
    """

    # Convert linear indices to multi-indices based on the original subarray shape
    multi_indices = np.unravel_index(linear_indices, shape)

    # Shift multi-indices by the offset to their position in the larger array
    if offset is not None:
        adjusted_multi_indices = tuple(m + o for m, o in zip(multi_indices, offset))
    else:
        adjusted_multi_indices = multi_indices

    # Convert back to linear indices in the larger ND array
    new_linear_indices = np.ravel_multi_index(adjusted_multi_indices, new_shape)

    return new_linear_indices


def update_csc_array_indices(sparse_mat, shape, new_shape, offset=None):
    """
    Update the row and column indices of a CSC-format sparse matrix to match a larger ND domain.

    Parameters:
    -----------
    sparse_mat : scipy.sparse.csc_array
        The input sparse matrix in CSC format.
    shape : tuple or ((tuple, tuple))
        The original shape of the subdomain.
    new_shape : tuple or ((tuple, tuple))
        The target shape of the larger domain.
    offset : tuple or ((tuple, tuple)), optional
        The offset of the subdomain within the larger domain (default: None).

    Returns:
    --------
    scipy.sparse.csc_array
        The updated sparse matrix with modified indices and shape.
    """

    # Extract matrix data and original indices
    data = sparse_mat.data
    row_indices = sparse_mat.indices  # Row indices (modifiable)
    col_pointers = sparse_mat.indptr  # Column pointers
    num_rows = sparse_mat.shape[0]  # Number of rows in the original matrix
    num_cols = sparse_mat.shape[1]  # Number of columns in the original matrix

    # Generate original linear row and column indices
    original_linear_rows = row_indices
    original_linear_cols = np.arange(num_cols)  # Columns as a sequential array

    # Translate row and column indices to the larger ND array
    shape = tuple(shape)
    if all(isinstance(dim, numbers.Integral) for dim in shape):
        shape = (shape, shape)
    new_shape = tuple(new_shape)
    if all(isinstance(dim, numbers.Integral) for dim in new_shape):
        new_shape = (new_shape, new_shape)
    if offset is None:
        offset = (None, None)
    else:
        offset = tuple(offset)
        if all(isinstance(dim, numbers.Integral) for dim in offset):
            offset = (offset, offset)

    if (shape[0] is None) or (new_shape[0] is None):
        new_row_indices = original_linear_rows
    else:
        new_row_indices = translate_indices_to_larger_array(
            original_linear_rows, shape[0], new_shape[0], offset[0]
        )
        num_rows = np.prod(new_shape[0])

    if (shape[1] is None) or (new_shape[1] is None):
        new_col_pointers = col_pointers
    else:
        new_col_indices = translate_indices_to_larger_array(
            original_linear_cols, shape[1], new_shape[1], offset[1]
        )
        num_cols = math.prod(new_shape[1])
        new_col_pointers = np.zeros(num_cols + 1, dtype=int)
        new_col_pointers[new_col_indices + 1] = np.diff(col_pointers)
        new_col_pointers = np.cumsum(new_col_pointers)

    # Create a new sparse matrix with the corrected 2D shape
    updated_mat = csc_array(
        (data, new_row_indices, new_col_pointers), shape=(num_rows, num_cols)
    )

    return updated_mat


def construct_interface_matrices(
    shapes,
    x_fs,
    x_cs=(None, None),
    ic=({"a": (1, 1), "b": (0, 0), "d": 0}, {"a": (0, 0), "b": (1, -1), "d": 0}),
    axis=0,
    shapes_d=(None, None),
):
    """
    Construct sparse matrices for computing interface conditions between two coupled subdomains.

    Parameters:
    -----------
    shapes : tuple of tuples
        Shapes of the two subdomains.
    x_fs : tuple of arrays
        Face-centered grid points for the two subdomains.
    x_cs : tuple of arrays, optional
        Cell-centered grid points for the two subdomains. If None, they are computed internally (default: (None, None)).
    ic : tuple of dicts, optional
        Interface conditions between the two subdomains, specified as:
        - `a`: Coefficients for the normal derivative terms.
        - `b`: Coefficients for the solution terms.
        - `d`: Source term at the interface.
        Defaults to conditions ensuring flux continuity.
    axis : int, optional
        The axis along which the interface is constructed (default: 0).

    Returns:
    --------
    interface_matrix_0 : scipy.sparse.csc_array
        Sparse matrix for computing interface values for the first subdomain.
    interface_bc_0 : scipy.sparse.csc_array
        Sparse matrix for boundary conditions in the first subdomain.
    interface_matrix_1 : scipy.sparse.csc_array
        Sparse matrix for computing interface values for the second subdomain.
    interface_bc_1 : scipy.sparse.csc_array
        Sparse matrix for boundary conditions in the second subdomain.

    Notes:
    ------
    This function constructs a fully implicit coupling between subdomains by ensuring the interface conditions
    are directly included in the global system of equations. It prevents the need for iterative coupling
    between domains, improving numerical stability.
    """

    if not all(
        s1 == s2 for i, (s1, s2) in enumerate(zip(shapes[0], shapes[1])) if i != axis
    ):
        raise ValueError(
            "Tuples shapes[0] and shapes[1] must be equal except for the specified axis."
        )
    shape = tuple(
        s1 + s2 if i == axis else s1
        for i, (s1, s2) in enumerate(zip(shapes[0], shapes[1]))
    )
    shape_i = tuple(1 if i == axis else s for i, s in enumerate(shape))

    # Extract the cell-centered grid points for the two subdomains
    for i in range(2):
        if x_cs[i] is None:
            x_cs = list(x_cs)
            x_cs[i] = 0.5 * (x_fs[i][1:] + x_fs[i][:-1])

    a, b = [
        [
            tuple(
                (
                    unwrap_bc_coeff(shape, ic_elem.get(key, (0, 0))[j], axis=axis)
                    if ic_elem and key in ic_elem
                    else np.zeros((1,) * len(shape_i))
                )
                for j in range(2)
            )
            for ic_elem in ic
        ]
        for key in ["a", "b"]
    ]

    d = [
        (
            unwrap_bc_coeff(shape, ic_elem.get("d", 0), axis=axis)
            if ic_elem and "d" in ic_elem
            else np.zeros((1,) * len(shape_i))
        )
        for ic_elem in ic
    ]

    alpha_1 = [None, None]
    alpha_1[0] = -(x_cs[0][-2] - x_fs[0][-1]) / (
        (x_cs[0][-1] - x_fs[0][-1]) * (x_cs[0][-2] - x_cs[0][-1])
    )
    alpha_1[1] = (x_cs[1][1] - x_fs[1][0]) / (
        (x_cs[1][0] - x_fs[1][0]) * (x_cs[1][1] - x_cs[1][0])
    )
    alpha_2 = [None, None]
    alpha_2[0] = -(x_cs[0][-1] - x_fs[0][-1]) / (
        (x_cs[0][-2] - x_fs[0][-1]) * (x_cs[0][-2] - x_cs[0][-1])
    )
    alpha_2[1] = (x_cs[1][0] - x_fs[1][0]) / (
        (x_cs[1][1] - x_fs[1][0]) * (x_cs[1][1] - x_cs[1][0])
    )
    alpha_0 = [alpha_1[0] - alpha_2[0], alpha_1[1] - alpha_2[1]]

    # reminder of notation
    # dc/dn[0] = alpha_0[0] c_i - alpha_1[0] c[-1] + alpha_2[0] c[-2]
    # dc/dn[1] = -alpha_0[1] c_i + alpha_1[1] c[0] - alpha_2[1] c[1]
    # interface conditions of the form:

    m = [[None for _ in range(2)] for _ in range(2)]
    v = [[None for _ in range(4)] for _ in range(2)]
    m_inv = [[None for _ in range(2)] for _ in range(2)]
    values = [[None for _ in range(4)] for _ in range(2)]
    for i in range(2):  # loop over the two conditions
        m[i][0] = a[i][0] * alpha_0[0] + b[i][0]
        m[i][1] = a[i][1] * alpha_0[1] + b[i][1]
        v[i][0] = -a[i][0] * alpha_2[0]
        v[i][1] = a[i][0] * alpha_1[0]
        v[i][2] = a[i][1] * alpha_1[1]
        v[i][3] = -a[i][1] * alpha_2[1]
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    det_inv = np.where(det != 0.0, 1.0 / det, 0.0)
    m_inv[0][0] = m[1][1] * det_inv
    m_inv[0][1] = -m[0][1] * det_inv
    m_inv[1][0] = -m[1][0] * det_inv
    m_inv[1][1] = m[0][0] * det_inv
    for j in range(2):
        for i in range(4):
            values[j][i] = m_inv[j][0] * v[0][i] + m_inv[j][1] * v[1][i]
            values[j][i] = np.broadcast_to(values[j][i], shape_i)
        values[j] = np.concatenate(values[j], axis=axis)

    shape_t = [math.prod(shape[0:axis]), shape[axis], math.prod(shape[axis + 1:])]
    n0, n1, n2 = shape_t
    i0 = np.arange(n0).reshape(-1, 1, 1)
    i2 = np.arange(n2).reshape(1, 1, -1)
    # Row indices: flat index into the interface (n0 x n2) array
    row_indices = np.ravel_multi_index(
        (i0, np.zeros((1, 4, 1), dtype=int), i2), (n0, 1, n2)
    )
    # Column indices: 4-point stencil around the interface
    stencil = (shapes[0][axis] + np.array([-2, -1, 0, 1])).reshape(1, -1, 1)
    col_indices = np.ravel_multi_index((i0, stencil, i2), shape_t)

    # Create the sparse matrix representing the interface
    interface_matrix = [None] * 2
    for j in range(2):
        fltr = values[j].ravel() != 0
        values_filtered = values[j].ravel()[fltr]
        row_indices_filtered = row_indices.ravel()[fltr]
        col_indices_filtered = col_indices.ravel()[fltr]
        interface_matrix[j] = csc_array(
            (values_filtered, (row_indices_filtered, col_indices_filtered)),
            shape=(shape_t[0] * shape_t[2], math.prod(shape_t)),
        )

    row_indices_bc = np.ravel_multi_index(
        (np.arange(n0).reshape(-1, 1), np.arange(n2).reshape(1, -1)), (n0, n2)
    )
    if shapes_d[0] is None and shapes_d[1] is None:
        interface_bc = [None] * 2
        for j in range(2):
            values_bc = m_inv[j][0] * d[0] + m_inv[j][1] * d[1]
            values_bc = np.broadcast_to(values_bc, shape_i)
            fltr = values_bc.ravel() != 0
            values_filtered = values_bc.ravel()[fltr]
            row_indices_filtered = row_indices_bc.ravel()[fltr]
            interface_bc[j] = csc_array(
                (values_filtered, row_indices_filtered, [0, row_indices_filtered.size]),
                shape=(shape_t[0] * shape_t[2], 1),
            )
        return (
            interface_matrix[0],
            interface_bc[0],
            interface_matrix[1],
            interface_bc[1],
        )
    else:
        interface_bc = [[None for _ in range(2)] for _ in range(2)]
        for j in range(2):
            for i in range(2):
                values_bc = m_inv[j][i] * d[i]
                values_bc = np.broadcast_to(values_bc, shape_i)
                fltr = values_bc.ravel() != 0
                values_filtered = values_bc.ravel()[fltr]
                row_indices_filtered = row_indices_bc.ravel()[fltr]
                if shapes_d[j] is None:
                    interface_bc[j][i] = csc_array(
                        (
                            values_filtered,
                            row_indices_filtered,
                            [0, row_indices_filtered.size],
                        ),
                        shape=(shape_t[0] * shape_t[2], 1),
                    )
                else:
                    num_cols = math.prod(shapes_d[j])
                    col_indices_bc = np.arange(num_cols, dtype=int).reshape(shapes_d[j])
                    col_indices_bc = np.broadcast_to(col_indices_bc, shape_i)
                    col_indices_filtered = col_indices_bc.ravel()[fltr]
                    interface_bc[j][i] = csc_array(
                        (values_filtered, (row_indices_filtered, col_indices_filtered)),
                        shape=(shape_t[0] * shape_t[2], num_cols),
                    )
        return (
            interface_matrix[0],
            interface_bc[0][0],
            interface_bc[0][1],
            interface_matrix[1],
            interface_bc[1][0],
            interface_bc[1][1],
        )
