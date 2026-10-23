"""Convective-flux operators and TVD limiter functions."""

import math
import numpy as np
from scipy.sparse import csc_array, csr_array
from .grid import generate_grid
from .interpolate import create_staggered_array, _align_to_axes
from .helpers import unwrap_bc_coeff, _sparse_array, substitute_outflow_bc


def construct_convflux_upwind(
    shape, x_f, x_c=None, bc=(None, None), v=1.0, axis=0, shapes_d=(None, None),
    format="csc"
):
    """Construct a first-order upwind convective-flux operator.

    Parameters
    ----------
    shape : tuple[int, ...] or int
        Cell-centered field shape.
    x_f : array_like
        Face coordinates along ``axis``.
    x_c : array_like, optional
        Cell-center coordinates. If omitted, arithmetic midpoints are used.
    bc : tuple[dict | None, dict | None], optional
        Left and right boundary-condition dictionaries with keys ``a``, ``b``,
        and ``d`` for ``a * dc/dn + b * c = d`` with ``n`` the outward normal
        (at the left end ``dc/dn = -dc/dx``). ``{"outflow": True}`` marks a pure-outflow boundary: the
        face value is the adjacent cell value (a stirred volume's exit). It is
        meant for faces where material leaves; if flow enters there, the face
        still carries the adjacent cell value.
    v : float or array_like, optional
        Face velocity field. Scalars and broadcastable arrays are accepted.
    axis : int, optional
        Convection axis.
    shapes_d : tuple[tuple | None, tuple | None], optional
        Optional source-vector shapes for boundary inhomogeneities. With
        ``shapes_d`` the dictionary's ``d`` is a coefficient on that external
        vector (use ``d = 1`` to pass the values through the vector).
    format : {'csc', 'csr'}, optional
        Sparse format for returned operator matrices.

    Returns
    -------
    tuple
        Without ``shapes_d``: ``(conv_matrix, conv_bc)``.
        With ``shapes_d``: ``(conv_matrix, conv_bc_left, conv_bc_right)``.
    """
    if isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)
    x_f, x_c = generate_grid(shape[axis], x_f, generate_x_c=True, x_c=x_c)

    v_f = create_staggered_array(v, shape, axis, x_f=x_f, x_c=x_c)
    conv_matrix = construct_convflux_upwind_int(shape, v_f, axis, format=format)
    if bc is None or bc == (None, None):
        shape_f = shape[:axis] + (shape[axis] + 1,) + shape[axis + 1:]
        conv_bc = csc_array((math.prod(shape_f), 1))
        return conv_matrix, conv_bc
    else:
        if shapes_d is None or shapes_d == (None, None):
            conv_matrix_bc, conv_bc = construct_convflux_bc(
                shape, x_f, x_c, bc, v_f, axis, format=format
            )
            conv_matrix += conv_matrix_bc
            return conv_matrix, conv_bc
        else:
            conv_matrix_bc_0, conv_bc_0, conv_matrix_bc_1, conv_bc_1 = (
                construct_convflux_bc(
                    shape, x_f, x_c, bc, v_f, axis, shapes_d=shapes_d, format=format
                )
            )
            conv_matrix += conv_matrix_bc_0 + conv_matrix_bc_1
            return conv_matrix, conv_bc_0, conv_bc_1


def construct_convflux_upwind_int(shape, v=1.0, axis=0, format="csc"):
    """Construct the internal-face upwind advection operator.

    Parameters
    ----------
    shape : tuple[int, ...]
        Cell-centered field shape.
    v : float or array_like, optional
        Face velocity field.
    axis : int, optional
        Convection axis.
    format : {'csc', 'csr'}, optional
        Sparse format of the returned matrix.

    Returns
    -------
    scipy.sparse.csc_array or scipy.sparse.csr_array
        Sparse matrix mapping cell-centered values to interior face fluxes.
    """
    shape_f = shape[:axis] + (shape[axis] + 1,) + shape[axis + 1:]
    shape_t = (math.prod(shape[:axis]), shape[axis], math.prod(shape[axis + 1:]))
    shape_f_t = (shape_t[0], shape_f[axis], shape_t[2])

    n0, n1, n2 = shape_t
    if isinstance(v, (float, int)):
        v_t = np.broadcast_to(np.array(v), shape_f_t)
    else:
        v_t = v.reshape(shape_f_t)
    fltr_v_pos = v_t > 0
    # Internal faces: indices 1 .. n1-1 along the axis
    i0 = np.arange(n0).reshape(-1, 1, 1)
    i1_int = np.arange(1, n1).reshape(1, -1, 1)
    i2 = np.arange(n2).reshape(1, 1, -1)
    i_f = np.ravel_multi_index((i0, i1_int, i2), shape_f_t)
    # Upwind: shift cell index left by 1 when velocity is positive
    i_c = np.ravel_multi_index((i0, i1_int - fltr_v_pos[:, 1:-1, :], i2), shape_t)
    if format == "csc":
        conv_matrix = csc_array(
            (v_t[:, 1:-1, :].ravel(), (i_f.ravel(), i_c.ravel())),
            shape=(math.prod(shape_f_t), math.prod(shape_t)),
        )
    elif format == "csr":
        # Each internal face (j=1..n1-1) has 1 entry; boundary faces have 0.
        nnz_per_face = np.zeros(n1 + 1, dtype=np.intp)
        nnz_per_face[1:n1] = 1
        row_nnz = np.tile(np.repeat(nnz_per_face, n2), n0)
        indptr = np.zeros(n0 * (n1 + 1) * n2 + 1, dtype=np.intp)
        np.cumsum(row_nnz, out=indptr[1:])
        conv_matrix = csr_array(
            (v_t[:, 1:-1, :].ravel(), i_c.ravel(), indptr),
            shape=(math.prod(shape_f_t), math.prod(shape_t)),
        )
    else:
        raise ValueError(
            f"format must be 'csc' or 'csr', got {format!r}"
        )
    conv_matrix.sort_indices()
    return conv_matrix


def construct_convflux_bc(
    shape, x_f, x_c=None, bc=(None, None), v=1.0, axis=0, shapes_d=(None, None),
    format="csc"
):
    """Construct boundary-face upwind corrections and source terms.

    Parameters
    ----------
    shape : tuple[int, ...]
        Cell-centered field shape.
    x_f : array_like
        Face coordinates along ``axis``.
    x_c : array_like, optional
        Cell-center coordinates.
    bc : tuple[dict | None, dict | None], optional
        Left and right boundary-condition dictionaries with keys ``a``, ``b``,
        and ``d`` for ``a * dc/dn + b * c = d`` with ``n`` the outward normal;
        ``{"outflow": True}`` marks a pure-outflow boundary.
    v : float or array_like, optional
        Face velocity field.
    axis : int, optional
        Convection axis.
    shapes_d : tuple[tuple | None, tuple | None], optional
        Optional source-vector shapes for inhomogeneous boundary terms; ``d``
        is then a coefficient on the external vector.
    format : {'csc', 'csr'}, optional
        Sparse format for returned operator matrices.

    Returns
    -------
    tuple
        ``(conv_matrix_bc, conv_bc)`` when ``shapes_d`` is not supplied, or
        ``(conv_matrix_left, conv_bc_left, conv_matrix_right, conv_bc_right)``
        otherwise.
    """

    # A pure-outflow face takes the value of the adjacent cell. The face values
    # are first built with a zero-gradient condition there (consistent with
    # construct_grad), then the outflow face is set to the adjacent cell value.
    bc, outflow = substitute_outflow_bc(bc, {"a": 1.0, "b": 0.0, "d": 0.0})

    # Trick: Reshape to triplet shape_t
    shape_f = shape[:axis] + (shape[axis] + 1,) + shape[axis + 1:]
    shape_t = (math.prod(shape[:axis]), shape[axis], math.prod(shape[axis + 1:]))
    shape_f_t = (shape_t[0], shape_f[axis], shape_t[2])
    shape_bc = shape[:axis] + (1,) + shape[axis + 1:]
    shape_bc_d = (shape_t[0], shape_t[2])

    n0, n1, n2 = shape_t
    i0 = np.arange(n0).reshape(-1, 1, 1)
    i2 = np.arange(n2).reshape(1, 1, -1)

    # Handle special case with one cell in the dimension axis
    if n1 == 1:
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
        if x_c is None:
            x_c = 0.5 * (x_f[0:-1] + x_f[1:])
        # Both faces reference cell 0
        i_c = np.ravel_multi_index(
            (i0, np.array([0, 0]).reshape(1, -1, 1), i2), shape_t
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
        fctr = (b[0] + alpha_0_left * a[0]) * (
            b[1] + alpha_0_right * a[1]
        ) - alpha_2_left * alpha_2_right * a[0] * a[1]
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        values = np.empty((shape_t[0], 2, shape_t[2]))
        values[:, 0, :] = np.broadcast_to(
            alpha_1 * a[0] * (a[1] * (alpha_0_right - alpha_2_left) + b[1]) * fctr,
            shape,
        ).reshape(shape_bc_d)
        values[:, 1, :] = np.broadcast_to(
            alpha_1 * a[1] * (a[0] * (alpha_0_left - alpha_2_right) + b[0]) * fctr,
            shape,
        ).reshape(shape_bc_d)

        i_f_bc = np.ravel_multi_index(
            (i0, np.array([0, shape_f_t[1] - 1]).reshape(1, -1, 1), i2), shape_f_t
        )

        def on_bc(value):
            return np.broadcast_to(value, shape_bc).reshape(shape_bc_d)

        # values_bc_src[:, face, source]: with one cell, each face value depends
        # on the data of both boundaries
        values_bc_src = np.zeros((n0, 2, 2, n2))
        values_bc_src[:, 0, 0, :] = on_bc((a[1] * alpha_0_right + b[1]) * d[0] * fctr)
        values_bc_src[:, 0, 1, :] = on_bc(-alpha_2_left * a[0] * d[1] * fctr)
        values_bc_src[:, 1, 1, :] = on_bc((a[0] * alpha_0_left + b[0]) * d[1] * fctr)
        values_bc_src[:, 1, 0, :] = on_bc(-alpha_2_right * a[1] * d[0] * fctr)
        for side in (0, 1):
            if outflow[side]:
                values[:, side, :], values_bc_src[:, side, :, :] = 1.0, 0.0
        values_bc = values_bc_src.sum(axis=2)

        if isinstance(v, (float, int)):
            values *= v
            values_bc *= v
            values_bc_src *= v
        else:
            slicer = [slice(None)] * len(shape)
            slicer[axis] = [0, -1]
            shape_f_b = list(shape_f)
            shape_f_b[axis] = 2
            values = values.reshape(shape_f_b)
            values *= v[tuple(slicer)]
            values_bc = values_bc.reshape(shape_f_b)
            values_bc *= v[tuple(slicer)]
            values_bc_src *= v[tuple(slicer)].reshape((n0, 2, 1, n2))
        conv_matrix = _sparse_array(
            (values.ravel(), (i_f.ravel(), i_c.ravel())),
            shape=(math.prod(shape_f_t), math.prod(shape_t)),
            format=format,
        )
    else:
        # Cell indices: 2 near left + 2 near right boundary
        cell_axis_idx = np.array([0, 1, n1 - 2, n1 - 1]).reshape(1, -1, 1)
        i_c = np.ravel_multi_index((i0, cell_axis_idx, i2), shape_t)
        # Face indices: left face (0) twice, right face twice
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
        a, b, d = [
            (
                unwrap_bc_coeff(shape, bc[0][key], axis=axis)
                if bc[0]
                else np.zeros((1,) * len(shape))
            )
            for key in ["a", "b", "d"]
        ]
        alpha_1 = (x_c[1] - x_f[0]) / ((x_c[0] - x_f[0]) * (x_c[1] - x_c[0]))
        alpha_2 = (x_c[0] - x_f[0]) / ((x_c[1] - x_f[0]) * (x_c[1] - x_c[0]))
        alpha_0 = alpha_1 - alpha_2
        fctr = alpha_0 * a + b
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        a_fctr = a * fctr
        a_fctr = np.broadcast_to(a_fctr, shape_bc).reshape(shape_bc_d)
        d_fctr = d * fctr
        d_fctr = np.broadcast_to(d_fctr, shape_bc).reshape(shape_bc_d)
        values[:, 0, :] = a_fctr * alpha_1
        values[:, 1, :] = -a_fctr * alpha_2
        values_bc[:, 0, :] = d_fctr

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
        fctr = alpha_0 * a + b
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        a_fctr = a * fctr
        a_fctr = np.broadcast_to(a_fctr, shape_bc).reshape(shape_bc_d)
        d_fctr = d * fctr
        d_fctr = np.broadcast_to(d_fctr, shape_bc).reshape(shape_bc_d)
        values[:, -1, :] = a_fctr * alpha_1
        values[:, -2, :] = -a_fctr * alpha_2
        values_bc[:, -1, :] = d_fctr
        if outflow[0]:
            values[:, 0, :], values[:, 1, :], values_bc[:, 0, :] = 1.0, 0.0, 0.0
        if outflow[1]:
            values[:, -1, :], values[:, -2, :], values_bc[:, -1, :] = 1.0, 0.0, 0.0
        if isinstance(v, (float, int)):
            values *= v
            values_bc *= v
        else:
            slicer = [slice(None)] * len(shape)
            slicer[axis] = [0, 0, -1, -1]
            shape_f_b = list(shape_f)
            shape_f_b[axis] = 4
            values = values.reshape(shape_f_b)
            values *= v[tuple(slicer)]
            shape_f_b[axis] = 2
            slicer[axis] = [0, -1]
            values_bc = values_bc.reshape(shape_f_b)
            values_bc *= v[tuple(slicer)]
        values_bc = values_bc.reshape((n0, 2, n2))
        values_bc_src = np.zeros((n0, 2, 2, n2))
        values_bc_src[:, 0, 0, :] = values_bc[:, 0, :]
        values_bc_src[:, 1, 1, :] = values_bc[:, 1, :]

    if (shapes_d[0] is None) and (shapes_d[1] is None):
        conv_bc = csc_array(
            (values_bc.ravel(), i_f_bc.ravel(), np.array([0, i_f_bc.size])),
            shape=(math.prod(shape_f_t), 1),
        )
        conv_matrix = _sparse_array(
            (values.ravel(), (i_f.ravel(), i_c.ravel())),
            shape=(math.prod(shape_f_t), math.prod(shape_t)),
            format=format,
        )
        conv_matrix.sort_indices()
        return conv_matrix, conv_bc
    else:
        values = values.reshape((shape_t[0], -1, shape_t[2]))
        conv_bc = [None] * 2
        shapes_d = list(shapes_d)
        for i in range(2):
            if shapes_d[i] is None:
                shapes_d[i] = (1,) * len(shape_bc)
            num_cols = math.prod(shapes_d[i])
            i_cols_bc = np.arange(num_cols, dtype=int).reshape(shapes_d[i])
            i_cols_bc = np.broadcast_to(i_cols_bc, shape_bc).reshape(n0, 1, n2)
            faces = [0, 1] if n1 == 1 else [i]
            conv_bc[i] = csc_array(
                (
                    values_bc_src[:, faces, i, :].ravel(),
                    (
                        i_f_bc[:, faces, :].ravel(),
                        np.broadcast_to(i_cols_bc, (n0, len(faces), n2)).ravel(),
                    ),
                ),
                shape=(math.prod(shape_f_t), num_cols),
            )
        if shape_t[1] == 1:
            conv_matrix_0 = _sparse_array(
                (values[:, 0, :].ravel(), (i_f[:, 0, :].ravel(), i_c[:, 0, :].ravel())),
                shape=(math.prod(shape_f_t), math.prod(shape_t)),
                format=format,
            )
            conv_matrix_1 = _sparse_array(
                (
                    values[:, -1, :].ravel(),
                    (i_f[:, -1, :].ravel(), i_c[:, -1, :].ravel()),
                ),
                shape=(math.prod(shape_f_t), math.prod(shape_t)),
                format=format,
            )
        else:
            conv_matrix_0 = _sparse_array(
                (
                    values[:, :2, :].ravel(),
                    (i_f[:, :2, :].ravel(), i_c[:, :2, :].ravel()),
                ),
                shape=(math.prod(shape_f_t), math.prod(shape_t)),
                format=format,
            )
            conv_matrix_1 = _sparse_array(
                (
                    values[:, -2:, :].ravel(),
                    (i_f[:, -2:, :].ravel(), i_c[:, -2:, :].ravel()),
                ),
                shape=(math.prod(shape_f_t), math.prod(shape_t)),
                format=format,
            )
        return conv_matrix_0, conv_bc[0], conv_matrix_1, conv_bc[1]


# TVD Limiters


def upwind(normalized_c_c, normalized_x_c, normalized_x_d):
    """Return zero correction (first-order upwind limiter)."""
    normalized_concentration_diff = np.zeros_like(normalized_c_c)
    return normalized_concentration_diff


def minmod(normalized_c_c, normalized_x_c, normalized_x_d):
    """Compute the Minmod TVD correction in normalized-variable space."""
    normalized_concentration_diff = np.maximum(
        0,
        (normalized_x_d - normalized_x_c)
        * np.minimum(
            normalized_c_c / normalized_x_c, (1 - normalized_c_c) / (1 - normalized_x_c)
        ),
    )
    return normalized_concentration_diff


def osher(normalized_c_c, normalized_x_c, normalized_x_d):
    """Compute the Osher TVD correction in normalized-variable space."""
    normalized_concentration_diff = np.maximum(
        0,
        np.where(
            normalized_c_c < normalized_x_c / normalized_x_d,
            (normalized_x_d / normalized_x_c - 1) * normalized_c_c,
            1 - normalized_c_c,
        ),
    )
    return normalized_concentration_diff


def clam(normalized_c_c, normalized_x_c, normalized_x_d):
    """Compute the CLAM TVD correction in normalized-variable space.

    CLAM, van Leer's curved-line advection method, is the parabola through
    (0, 0), (x_c, x_d) and (1, 1); on a uniform grid ``c_f = 2 c_c - c_c**2``.
    On a uniform grid this curve is also van Leer's harmonic limiter, and
    pymrm uses the same parabola, with the same cap at ``c_f = 1``, for both:
    ``clam`` and ``vanleer`` agree on every stencil. Before pymrm 2.5.0
    ``clam`` returned the Osher curve.
    """
    return vanleer(normalized_c_c, normalized_x_c, normalized_x_d)


def muscl(normalized_c_c, normalized_x_c, normalized_x_d):
    """Compute the MUSCL TVD correction in normalized-variable space.

    Uniform grid: 2 c_c up to 1/4, c_c + 1/4 up to 3/4, then 1.
    """
    normalized_concentration_diff = np.maximum(
        0,
        np.where(
            # the first line meets c_c + x_d - x_c at c_c = x_c / 2
            normalized_c_c < normalized_x_c / 2,
            ((2 * normalized_x_d - normalized_x_c) / normalized_x_c - 1)
            * normalized_c_c,  # noqa: E501
            np.where(
                normalized_c_c < 1 + normalized_x_c - normalized_x_d,
                normalized_x_d - normalized_x_c,
                1 - normalized_c_c,
            ),
        ),
    )  # noqa: E501
    return normalized_concentration_diff


def smart(normalized_c_c, normalized_x_c, normalized_x_d):
    """Compute the SMART TVD correction in normalized-variable space."""
    normalized_concentration_diff = np.maximum(
        0,
        np.where(
            normalized_c_c < normalized_x_c / 3,
            (
                normalized_x_d
                * (1 - 3 * normalized_x_c + 2 * normalized_x_d)
                / (normalized_x_c * (1 - normalized_x_c))
                - 1
            )
            * normalized_c_c,  # noqa: E501
            np.where(
                normalized_c_c
                < normalized_x_c
                / normalized_x_d
                * (1 + normalized_x_d - normalized_x_c),  # noqa: E501
                (
                    normalized_x_d * (normalized_x_d - normalized_x_c)  # noqa: E501
                    + normalized_x_d
                    * (1 - normalized_x_d)
                    / normalized_x_c
                    * normalized_c_c
                )
                / (1 - normalized_x_c)
                - normalized_c_c,
                1 - normalized_c_c,
            ),
        ),
    )  # noqa: E501
    return normalized_concentration_diff


def stoic(normalized_c_c, normalized_x_c, normalized_x_d):
    """Compute the STOIC TVD correction in normalized-variable space.

    Piecewise: the SMART line ``c_f = k c_c`` up to its intersection with the
    central-difference line, then central differencing up to ``c_c = x_c``,
    then QUICK, then ``c_f = 1``. On a uniform grid (``x_c = 1/2``,
    ``x_d = 3/4``) this is 3 c_c, (1 + c_c)/2, 3/8 + 3 c_c/4 and 1, with breaks
    at 1/5, 1/2 and 5/6.
    """
    x_c = normalized_x_c
    x_d = normalized_x_d
    c_c = normalized_c_c
    slope = x_d * (1 - 3 * x_c + 2 * x_d) / (x_c * (1 - x_c))
    # intersection of c_f = slope * c_c with the central-difference line
    c_break = x_c / (1 + 2 * x_d)
    c_f = np.where(
        c_c < c_break,
        slope * c_c,
        np.where(
            c_c < x_c,
            (x_d - x_c + (1 - x_d) * c_c) / (1 - x_c),
            np.where(
                c_c < x_c / x_d * (1 + x_d - x_c),
                (x_d * (x_d - x_c) + x_d * (1 - x_d) / x_c * c_c) / (1 - x_c),
                1.0,
            ),
        ),
    )
    normalized_concentration_diff = np.maximum(0, c_f - c_c)
    return normalized_concentration_diff


def vanleer(normalized_c_c, normalized_x_c, normalized_x_d):
    """Compute the van-Leer TVD correction in normalized-variable space.

    The curve is the parabola through (0, 0), (x_c, x_d) and (1, 1), which on
    a uniform grid is van Leer's harmonic limiter and also his CLAM curve
    (``c_f = 2 c_c - c_c**2``); ``clam`` returns the same values. It rises
    above ``c_f = 1`` when ``x_d - x_c > x_c (1 - x_c)``, for example next to a
    boundary where the upstream point is a face half a cell away (x_c = 1/3,
    x_d = 2/3); the correction is capped at ``1 - c_c`` to stay bounded. The
    cap acts only above c_c = x_c and does not reduce the order.
    """
    normalized_concentration_diff = np.maximum(
        0,
        np.minimum(
            normalized_c_c
            * (1 - normalized_c_c)
            * (normalized_x_d - normalized_x_c)
            / (normalized_x_c * (1 - normalized_x_c)),
            1 - normalized_c_c,
        ),
    )
    return normalized_concentration_diff


# Derivatives of the limiters with respect to the normalized cell value, at
# fixed normalized positions. Each follows the branch its limiter selects, so a
# value and its derivative always come from the same piece; where the limiter
# is clipped to zero the derivative is zero.


def _upwind_derivative(normalized_c_c, normalized_x_c, normalized_x_d):
    return np.zeros_like(normalized_c_c)


def _minmod_derivative(normalized_c_c, normalized_x_c, normalized_x_d):
    c_c, x_c, x_d = normalized_c_c, normalized_x_c, normalized_x_d
    lower = c_c / x_c <= (1 - c_c) / (1 - x_c)
    slope = (x_d - x_c) * np.where(lower, 1 / x_c, -1 / (1 - x_c))
    return np.where(minmod(c_c, x_c, x_d) > 0, slope, 0.0)


def _osher_derivative(normalized_c_c, normalized_x_c, normalized_x_d):
    c_c, x_c, x_d = normalized_c_c, normalized_x_c, normalized_x_d
    slope = np.where(c_c < x_c / x_d, x_d / x_c - 1, -1.0)
    return np.where(osher(c_c, x_c, x_d) > 0, slope, 0.0)


def _muscl_derivative(normalized_c_c, normalized_x_c, normalized_x_d):
    c_c, x_c, x_d = normalized_c_c, normalized_x_c, normalized_x_d
    slope = np.where(
        c_c < x_c / 2,
        (2 * x_d - x_c) / x_c - 1,
        np.where(c_c < 1 + x_c - x_d, 0.0, -1.0),
    )
    return np.where(muscl(c_c, x_c, x_d) > 0, slope, 0.0)


def _smart_derivative(normalized_c_c, normalized_x_c, normalized_x_d):
    c_c, x_c, x_d = normalized_c_c, normalized_x_c, normalized_x_d
    slope = np.where(
        c_c < x_c / 3,
        x_d * (1 - 3 * x_c + 2 * x_d) / (x_c * (1 - x_c)) - 1,
        np.where(
            c_c < x_c / x_d * (1 + x_d - x_c),
            x_d * (1 - x_d) / (x_c * (1 - x_c)) - 1,
            -1.0,
        ),
    )
    return np.where(smart(c_c, x_c, x_d) > 0, slope, 0.0)


def _stoic_derivative(normalized_c_c, normalized_x_c, normalized_x_d):
    c_c, x_c, x_d = normalized_c_c, normalized_x_c, normalized_x_d
    slope = np.where(
        c_c < x_c / (1 + 2 * x_d),
        x_d * (1 - 3 * x_c + 2 * x_d) / (x_c * (1 - x_c)),
        np.where(
            c_c < x_c,
            (1 - x_d) / (1 - x_c),
            np.where(
                c_c < x_c / x_d * (1 + x_d - x_c),
                x_d * (1 - x_d) / (x_c * (1 - x_c)),
                0.0,
            ),
        ),
    )
    return np.where(stoic(c_c, x_c, x_d) > 0, slope - 1, 0.0)


def _vanleer_derivative(normalized_c_c, normalized_x_c, normalized_x_d):
    c_c, x_c, x_d = normalized_c_c, normalized_x_c, normalized_x_d
    scale = (x_d - x_c) / (x_c * (1 - x_c))
    parabola = c_c * (1 - c_c) * scale <= 1 - c_c
    slope = np.where(parabola, (1 - 2 * c_c) * scale, -1.0)
    return np.where(vanleer(c_c, x_c, x_d) > 0, slope, 0.0)


upwind.derivative = _upwind_derivative
minmod.derivative = _minmod_derivative
osher.derivative = _osher_derivative
clam.derivative = _vanleer_derivative
muscl.derivative = _muscl_derivative
smart.derivative = _smart_derivative
stoic.derivative = _stoic_derivative
vanleer.derivative = _vanleer_derivative


def _limiter_derivative(tvd_limiter, normalized_c_c, normalized_x_c, normalized_x_d, step=1e-6):
    """Derivative of a limiter: its ``derivative`` attribute, else a central difference."""
    derivative = getattr(tvd_limiter, "derivative", None)
    if derivative is not None:
        return derivative(normalized_c_c, normalized_x_c, normalized_x_d)
    return (
        tvd_limiter(normalized_c_c + step, normalized_x_c, normalized_x_d)
        - tvd_limiter(normalized_c_c - step, normalized_x_c, normalized_x_d)
    ) / (2 * step)


class ConvFluxTVD:
    """TVD convective face flux with its exact Jacobians, for Newton solves.

    Build the object once per axis, then call it with the current state in
    every residual evaluation. It returns the face fluxes ``v * c_f``, with
    ``c_f`` the TVD face values of :func:`interp_cntr_to_stagg_tvd`, and the
    derivative of those fluxes with respect to the state: the upwind part, the
    limiter part and the boundary faces together. A steady problem solved with
    this Jacobian converges in a few Newton steps for every limiter; deferred
    correction (the TVD correction in the residual, the upwind Jacobian)
    converges only slowly with minmod and stalls with van Leer, MUSCL and
    SMART (pitfalls P10).

    Parameters
    ----------
    shape : tuple[int, ...] or int
        Cell-centered field shape.
    x_f : array_like
        Face coordinates along ``axis``.
    x_c : array_like, optional
        Cell-center coordinates. If omitted, arithmetic midpoints are used.
    bc : tuple[dict | None, dict | None], optional
        Left and right boundary conditions ``a * dc/dn + b * c = d`` with
        ``n`` the outward normal, or ``{"outflow": True}``, as for
        :func:`construct_convflux_upwind`. Boundary faces carry the boundary
        value without a TVD correction.
    tvd_limiter : callable, optional
        Limiter ``phi(c_norm, x_norm_c, x_norm_d)`` (default :func:`minmod`).
        The built-in limiters carry an analytic ``derivative`` attribute; for
        any other limiter the derivative is a central difference in
        ``c_norm``.
    axis : int, optional
        Convection axis.
    shapes_d : tuple[tuple | None, tuple | None], optional
        Shapes of external vectors for the boundary data, as for
        :func:`construct_convflux_upwind`: ``d`` is then a coefficient on that
        vector, and the call takes the vectors and also returns the Jacobians
        with respect to them.
    format : {'csc', 'csr'}, optional
        Sparse format of the returned Jacobians.

    Notes
    -----
    Call ``flux, jac = tvd(c, v)``; with ``shapes_d``,
    ``flux, jac, jac_bc_left, jac_bc_right = tvd(c, v, d=(ext_left, ext_right))``;
    ``jac_v=True`` appends the Jacobian with respect to ``v``. See
    :meth:`__call__`.

    The face upwind of the first interior face is the boundary value, so the
    Jacobian there reaches the cells that boundary value depends on, and the
    boundary data. The flow direction of each interior face follows
    ``v > 0``; an initial guess with ``v = 0`` everywhere gives a zero
    convective Jacobian. Where ``c_D == c_U`` the face is upwind and its
    derivative is the upwind one (the built-in limiters are also zero outside
    ``0 < c_norm < 1``). Use ``newton(..., damping="monotone")``: undamped Newton can cycle
    between the pieces of MUSCL, SMART and STOIC. The
    sparsity pattern covers both flow directions, so it does not change when
    ``v`` changes sign; it contains explicit zeros, which sparse products drop.
    """

    def __init__(
        self, shape, x_f, x_c=None, bc=(None, None), tvd_limiter=minmod, axis=0,
        shapes_d=(None, None), format="csc"
    ):
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        if axis < 0:
            axis += len(shape)
        x_f, x_c = generate_grid(shape[axis], x_f, generate_x_c=True, x_c=x_c)
        if bc is None:
            bc = (None, None)
        self.shape = shape
        self.axis = axis
        self.shape_f = shape[:axis] + (shape[axis] + 1,) + shape[axis + 1:]
        self._x_f, self._x_c = x_f, x_c
        self.tvd_limiter = upwind if tvd_limiter is None else tvd_limiter
        self.format = format
        n0, n1, n2 = math.prod(shape[:axis]), shape[axis], math.prod(shape[axis + 1:])
        self._shape_t = (n0, n1, n2)
        self._shape_f_t = (n0, n1 + 1, n2)

        if shapes_d is None:
            shapes_d = (None, None)
        self.shapes_d = tuple(shapes_d)
        self._with_d = any(s is not None for s in self.shapes_d)
        ones = (1,) * len(shape)
        shapes_ext = tuple(ones if s is None else tuple(s) for s in self.shapes_d)
        self._n_ext = tuple(math.prod(s) for s in shapes_ext)

        # Boundary faces: value maps with the outflow face set to its cell. As
        # the upwind point of the neighbouring face, an outflow boundary takes
        # the zero-gradient value, as in interp_cntr_to_stagg_tvd.
        bc_upwind, _ = substitute_outflow_bc(bc, {"a": 1.0, "b": 0.0, "d": 0.0})
        m_0, e_0, m_1, e_1 = construct_convflux_bc(
            shape, x_f, x_c, bc, 1.0, axis, shapes_d=shapes_ext, format="csr")
        self._m_face, self._e_face = (m_0 + m_1).tocsr(), (e_0.tocsr(), e_1.tocsr())
        m_0, e_0, m_1, e_1 = construct_convflux_bc(
            shape, x_f, x_c, bc_upwind, 1.0, axis, shapes_d=shapes_ext, format="csr")
        self._m_up, self._e_up = (m_0 + m_1).tocsr(), (e_0.tocsr(), e_1.tocsr())

        def coefficient(matrix, cell):
            """Coefficient of cell ``cell`` (along the axis) in each face row."""
            indicator = np.zeros(self._shape_t)
            indicator[:, cell, :] = 1.0
            return (matrix @ indicator.ravel()).reshape(self._shape_f_t)

        # band[:, face, o + 2, :]: dependence of a boundary face on cell face + o
        self._offsets = np.arange(-2, 2)
        self._band_bnd = np.zeros((n0, 2, 4, n2))
        for side, face in enumerate((0, n1)):
            for o in self._offsets:
                if 0 <= face + o < n1:
                    self._band_bnd[:, side, o + 2, :] = coefficient(self._m_face, face + o)[:, face, :]
        if n1 > 1:
            self._up_left = [coefficient(self._m_up, j)[:, 0, :] for j in (0, 1)]
            self._up_right = [coefficient(self._m_up, j)[:, n1, :] for j in (n1 - 1, n1 - 2)]

            # normalized positions of the interior faces 1 .. n1-1, per direction
            x_d = x_f[1:-1]
            x_U = np.concatenate((x_f[:1], x_c[:-2]))
            x_norm = {}
            for positive, x_up, x_C, x_D in (
                (True, x_U, x_c[:-1], x_c[1:]),
                (False, np.concatenate((x_c[2:], x_f[-1:])), x_c[1:], x_c[:-1]),
            ):
                x_norm[positive] = (
                    ((x_C - x_up) / (x_D - x_up)).reshape(1, -1, 1),
                    ((x_d - x_up) / (x_D - x_up)).reshape(1, -1, 1),
                )
            self._x_norm = x_norm

        # fixed sparsity pattern: face k, cell k + o, for o = -2 .. 1
        k = np.arange(n1 + 1).reshape(1, -1, 1, 1)
        cells = k + self._offsets.reshape(1, 1, -1, 1)
        valid = (cells >= 0) & (cells < n1)
        self._valid = np.broadcast_to(valid, (n0, n1 + 1, 4, n2))
        i0 = np.arange(n0).reshape(-1, 1, 1, 1)
        i2 = np.arange(n2).reshape(1, 1, 1, -1)
        rows = ((i0 * (n1 + 1) + k) * n2 + i2) + 0 * cells
        cols = ((i0 * n1 + np.clip(cells, 0, n1 - 1)) * n2 + i2)
        rows, cols = rows[self._valid], cols[self._valid]
        shape_jac = (math.prod(self._shape_f_t), math.prod(self._shape_t))
        tag = _sparse_array((np.arange(1, rows.size + 1, dtype=float), (rows, cols)),
                            shape=shape_jac, format=format)
        tag.sort_indices()
        self._perm = tag.data.astype(np.intp) - 1
        self._indices, self._indptr, self._shape_jac = tag.indices, tag.indptr, shape_jac

        # boundary faces and the interior faces next to them, as flat face indices
        f_t = np.arange(math.prod(self._shape_f_t)).reshape(self._shape_f_t)
        self._faces_left, self._faces_right = f_t[:, 0, :], f_t[:, n1, :]
        if n1 > 1:
            self._faces_second, self._faces_penult = f_t[:, 1, :], f_t[:, n1 - 1, :]

    def __call__(self, c, v, d=None, jac_v=False):
        """Evaluate the face fluxes and their Jacobians.

        Parameters
        ----------
        c : array_like
            Cell-centered values, any array with ``prod(shape)`` elements.
        v : float or array_like
            Face velocity (or any face weight, such as ``rho * cp * v``),
            accepted as by :func:`construct_convflux_upwind` (through
            :func:`create_staggered_array`): a scalar, a face-shaped or
            broadcastable array, or a 1-D array along the convection axis.
            Cell-centred values are interpolated to the faces, but not with
            ``jac_v=True``. Its sign selects the stencil of each interior face.
        d : tuple, optional
            With ``shapes_d``: the external vectors ``(ext_left, ext_right)``
            (``None`` for a side without a shape). Required then, ignored
            otherwise.
        jac_v : bool, optional
            Also return the Jacobian with respect to ``v``.

        Returns
        -------
        tuple
            ``(flux, jac)``; with ``shapes_d``
            ``(flux, jac, jac_bc_left, jac_bc_right)``; with ``jac_v=True``
            the Jacobian with respect to ``v`` is appended. ``flux`` has the
            face shape; apply :func:`construct_div` to ``flux.reshape(-1)``.
            ``jac`` is ``(n_faces, n_cells)``: use ``div @ jac``.
            ``jac_bc_*`` are ``(n_faces, prod(shapes_d[i]))`` derivatives only:
            ``flux`` already contains the boundary data. The Jacobian with
            respect to ``v`` is ``(n_faces, v.size)`` with its columns in the
            order of ``v``; its entries are the face values ``c_f``. For a
            velocity from a pressure gradient, chain it with that gradient
            matrix.
        """
        n0, n1, n2 = self._shape_t
        c = np.asarray(c, dtype=float).reshape(self._shape_t)
        v_in = np.asarray(v, dtype=float)
        try:
            v_aligned = _align_to_axes(v_in, self.shape, self.axis)
            v_t = create_staggered_array(v_in, self.shape, self.axis, x_f=self._x_f, x_c=self._x_c)
        except ValueError:
            raise ValueError(
                f"v with shape {v_in.shape} does not fit the face shape {self.shape_f}"
            ) from None
        centred = (v_in.size > 1 and self.shape[self.axis] > 1 and v_aligned.ndim == len(self.shape)
                   and v_aligned.shape[self.axis] == self.shape[self.axis])
        if jac_v and centred:
            raise ValueError("jac_v=True needs v on the faces, not at the cell centres")
        v_t = v_t.reshape(self._shape_f_t)
        ext = [np.ones(1), np.ones(1)]
        if self._with_d:
            if d is None:
                raise ValueError("built with shapes_d: pass d=(ext_left, ext_right)")
            for i in range(2):
                if self.shapes_d[i] is not None:
                    ext[i] = np.asarray(d[i], dtype=float).reshape(-1)
                    if ext[i].size != self._n_ext[i]:
                        raise ValueError(f"d[{i}] has {ext[i].size} values, shapes_d[{i}] needs {self._n_ext[i]}")

        c_flat = c.ravel()
        face_bnd = (self._m_face @ c_flat + self._e_face[0] @ ext[0]
                    + self._e_face[1] @ ext[1]).reshape(self._shape_f_t)
        face = np.zeros(self._shape_f_t)
        face[:, 0, :], face[:, n1, :] = face_bnd[:, 0, :], face_bnd[:, n1, :]
        band = np.zeros((n0, n1 + 1, 4, n2))       # d c_f / d c, before the velocity
        band[:, 0], band[:, n1] = self._band_bnd[:, 0], self._band_bnd[:, 1]
        g_left = g_right = None

        if n1 > 1:
            up = (self._m_up @ c_flat + self._e_up[0] @ ext[0]
                  + self._e_up[1] @ ext[1]).reshape(self._shape_f_t)
            pos = v_t[:, 1:-1, :] > 0
            c_U = np.where(pos, np.concatenate((up[:, :1], c[:, :-2]), axis=1),
                           np.concatenate((c[:, 2:], up[:, n1:]), axis=1))
            c_C = np.where(pos, c[:, :-1], c[:, 1:])
            c_D = np.where(pos, c[:, 1:], c[:, :-1])
            x_norm_c = np.where(pos, self._x_norm[True][0], self._x_norm[False][0])
            x_norm_d = np.where(pos, self._x_norm[True][1], self._x_norm[False][1])
            dc_DU = c_D - c_U
            c_norm = np.zeros_like(c_C)
            np.divide(c_C - c_U, dc_DU, out=c_norm, where=(dc_DU != 0))
            phi = self.tvd_limiter(c_norm, x_norm_c, x_norm_d)
            face[:, 1:-1, :] = c_C + phi * dc_DU
            active = dc_DU != 0
            dphi = np.zeros_like(c_norm)
            if np.any(active):
                dphi[active] = _limiter_derivative(
                    self.tvd_limiter, c_norm[active], x_norm_c[active], x_norm_d[active])
            w_C = 1 + dphi
            w_D = phi - c_norm * dphi
            w_U = -phi + (c_norm - 1) * dphi

            inner = band[:, 1:-1]                    # faces 1 .. n1-1, offsets -2 .. 1
            neg = ~pos
            # v > 0: U = k-2, C = k-1, D = k; v <= 0: U = k+1, C = k, D = k-1
            inner[:, :, 1] += np.where(pos, w_C, w_D)
            inner[:, :, 2] += np.where(pos, w_D, w_C)
            inner[:, 1:, 0] += np.where(pos[:, 1:], w_U[:, 1:], 0.0)
            inner[:, :-1, 3] += np.where(neg[:, :-1], w_U[:, :-1], 0.0)
            # the upwind point of face 1 (v > 0) and face n1-1 (v <= 0) is a boundary value
            g_left = np.where(pos[:, 0], w_U[:, 0], 0.0)
            inner[:, 0, 1] += g_left * self._up_left[0]
            inner[:, 0, 2] += g_left * self._up_left[1]
            g_right = np.where(neg[:, -1], w_U[:, -1], 0.0)
            inner[:, -1, 2] += g_right * self._up_right[0]
            inner[:, -1, 1] += g_right * self._up_right[1]

        flux = (v_t * face).reshape(self.shape_f)
        data = (band * v_t[:, :, None, :])[self._valid][self._perm]
        jac = _sparse_array((data, self._indices.copy(), self._indptr.copy()),
                            shape=self._shape_jac, format=self.format)
        out = [flux, jac]

        if self._with_d:
            n_f = math.prod(self._shape_f_t)
            v_flat = v_t.ravel()
            on_bnd = np.zeros(n_f)
            on_bnd[self._faces_left.ravel()] = 1.0
            on_bnd[self._faces_right.ravel()] = 1.0
            scale_bnd = csr_array((v_flat * on_bnd, (np.arange(n_f), np.arange(n_f))), shape=(n_f, n_f))
            for i, g, rows, cols in (
                (0, g_left, "_faces_second", "_faces_left"),
                (1, g_right, "_faces_penult", "_faces_right"),
            ):
                jac_bc = scale_bnd @ self._e_face[i]
                if g is not None:
                    r, k = getattr(self, rows).ravel(), getattr(self, cols).ravel()
                    chain = csr_array(((v_flat[r] * g.ravel()), (r, k)), shape=(n_f, n_f))
                    jac_bc = jac_bc + chain @ self._e_up[i]
                out.append(jac_bc.asformat(self.format))

        if jac_v:
            n_f = math.prod(self._shape_f_t)
            v_cols = create_staggered_array(np.arange(v_in.size).reshape(v_in.shape), self.shape,
                                            self.axis, x_f=self._x_f, x_c=self._x_c).ravel()
            out.append(_sparse_array((face.ravel(), (np.arange(n_f), v_cols)),
                                     shape=(n_f, v_in.size), format=self.format))
        return tuple(out)
