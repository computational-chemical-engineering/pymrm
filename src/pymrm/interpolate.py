"""Interpolation utilities between cell-centered and staggered grids."""

import math
import numpy as np
from scipy.sparse import csc_array, csr_array
from .helpers import (
    substitute_outflow_bc, is_outflow_bc, _field_geometry,
    _single_cell_system, _wall_alphas, _on_bc, _wall_factor)


def _geometry_along(shape, axis):
    """Field geometry of ``shape`` along ``axis``, which may be negative."""
    return _field_geometry(shape, axis + len(shape) if axis < 0 else axis)


def interp_stagg_to_cntr(staggered_values, x_f, x_c=None, axis=0):
    """Interpolate face/staggered values to cell centers.

    Parameters
    ----------
    staggered_values : numpy.ndarray
        Values defined on staggered (face) locations.
    x_f : array_like
        Face coordinates along ``axis``.
    x_c : array_like, optional
        Cell-center coordinates. If omitted, midpoint interpolation is used.
    axis : int, optional
        Interpolation axis.

    Returns
    -------
    numpy.ndarray
        Cell-centered values with one fewer element along ``axis``.
    """
    shape_f = list(staggered_values.shape)
    if axis < 0:
        axis += len(shape_f)
    shape_f_t = [
        math.prod(shape_f[:axis]),
        math.prod(shape_f[axis: axis + 1]),
        math.prod(shape_f[axis + 1:]),
    ]
    shape = shape_f.copy()
    shape[axis] -= 1
    staggered_values = np.reshape(staggered_values, shape_f_t)

    if x_c is None:
        cell_centered_values = 0.5 * (
            staggered_values[:, 1:, :] + staggered_values[:, :-1, :]
        )
    else:
        wght = (x_c - x_f[:-1]) / (x_f[1:] - x_f[:-1])
        cell_centered_values = staggered_values[:, :-1, :] + wght.reshape(
            (1, -1, 1)
        ) * (staggered_values[:, 1:, :] - staggered_values[:, :-1, :])

    return cell_centered_values.reshape(shape)


def interp_cntr_to_stagg(cell_centered_values, x_f, x_c=None, axis=0):
    """Interpolate cell-centered values to face/staggered locations.

    Parameters
    ----------
    cell_centered_values : numpy.ndarray
        Values defined at cell centers.
    x_f : array_like
        Face coordinates along ``axis``.
    x_c : array_like, optional
        Cell-center coordinates. If omitted, midpoint locations are used.
    axis : int, optional
        Interpolation axis.

    Returns
    -------
    numpy.ndarray
        Staggered values with one additional element along ``axis``.
    """
    geo = _geometry_along(cell_centered_values.shape, axis)
    shape_t, shape_f, shape_f_t = geo.shape_t, geo.shape_f, geo.shape_f_t
    if x_c is None:
        x_c = 0.5 * (x_f[:-1] + x_f[1:])

    wght = (x_f[1:-1] - x_c[:-1]) / (x_c[1:] - x_c[:-1])
    cell_centered_values = cell_centered_values.reshape(shape_t)
    if shape_t[1] == 1:
        staggered_values = np.tile(cell_centered_values, (1, 2, 1))
    else:
        staggered_values = np.empty(shape_f_t)
        staggered_values[:, 1:-1, :] = cell_centered_values[:, :-1, :] + wght.reshape(
            (1, -1, 1)
        ) * (cell_centered_values[:, 1:, :] - cell_centered_values[:, :-1, :])
        staggered_values[:, 0, :] = (
            cell_centered_values[:, 0, :] * (x_c[1] - x_f[0])
            - cell_centered_values[:, 1, :] * (x_c[0] - x_f[0])
        ) / (x_c[1] - x_c[0])
        staggered_values[:, -1, :] = (
            cell_centered_values[:, -1, :] * (x_f[-1] - x_c[-2])
            - cell_centered_values[:, -2, :] * (x_f[-1] - x_c[-1])
        ) / (x_c[-1] - x_c[-2])
    return staggered_values.reshape(shape_f)


def _overwrite_outflow_faces(face, delta, cell_centered_values, outflow, axis):
    """Set the outflow faces to the adjacent cell value with zero correction."""
    cells = np.asarray(cell_centered_values)
    ax = axis % cells.ndim
    shape_f = list(cells.shape)
    shape_f[ax] += 1
    face = np.asarray(face).reshape(shape_f).copy()
    delta = np.asarray(delta).reshape(shape_f).copy()
    face, delta = np.moveaxis(face, ax, 0), np.moveaxis(delta, ax, 0)
    cells = np.moveaxis(cells, ax, 0)
    for flag, (i_face, i_cell) in zip(outflow, ((0, 0), (-1, -1))):
        if flag:
            face[i_face] = cells[i_cell]
            delta[i_face] = 0.0
    return np.moveaxis(face, 0, ax), np.moveaxis(delta, 0, ax)


def _on_boundary(value, geo):
    """``value`` broadcast over the boundary shape and flattened to ``(n0, n2)``.

    Unlike ``_on_bc`` this returns a writable copy (adding zeros turns ``-0.0`` into
    ``0.0``), which the face values built from it rely on to stay bit-identical.
    """
    return np.reshape(value + np.zeros(geo.shape_bc), geo.shape_bc_d)


def _interp_single_cell(cells, x_f, x_c, bc, geo):
    """Face values of a single-cell domain from the two boundary conditions."""
    staggered_values = np.empty(geo.shape_f_t)
    a, b, d, alpha_1, alpha_2, alpha_0, fctr = _single_cell_system(geo, x_f, x_c, bc)
    fctr_m = alpha_1 * a[0] * (a[1] * (alpha_0[1] - alpha_2[0]) + b[1]) * fctr
    staggered_values[:, 0, :] = _on_boundary(fctr_m, geo) * cells[:, 0, :]
    fctr_m = alpha_1 * a[1] * (a[0] * (alpha_0[0] - alpha_2[1]) + b[0]) * fctr
    staggered_values[:, 1, :] = _on_boundary(fctr_m, geo) * cells[:, 0, :]
    fctr_m = (
        (a[1] * alpha_0[1] + b[1]) * d[0] - alpha_2[0] * a[0] * d[1]
    ) * fctr
    staggered_values[:, 0, :] += _on_boundary(fctr_m, geo)
    fctr_m = (
        (a[0] * alpha_0[0] + b[0]) * d[1] - alpha_2[1] * a[1] * d[0]
    ) * fctr
    staggered_values[:, 1, :] += _on_boundary(fctr_m, geo)
    return staggered_values.reshape(geo.shape_f), np.zeros(geo.shape_f)


def _wall_face_value(bc_i, geo, alphas, c_near, c_far):
    """Face value at a boundary from the two adjacent cells and its boundary condition."""
    alpha_1, alpha_2 = alphas
    a, _, d, _, fctr = _wall_factor(bc_i, geo, alpha_1, alpha_2)
    a_fctr = _on_boundary(a * fctr, geo)
    d_fctr = _on_boundary(d * fctr, geo)
    return d_fctr + a_fctr * (alpha_1 * c_near - alpha_2 * c_far)


def _wall_faces(cells, x_f, x_c, bc, geo):
    """Face buffer of a multi-cell domain with the two boundary faces filled in."""
    faces = np.empty(geo.shape_f_t)
    left, right = _wall_alphas(x_f, x_c)
    faces[:, 0, :] = _wall_face_value(bc[0], geo, left, cells[:, 0, :], cells[:, 1, :])
    faces[:, -1, :] = _wall_face_value(bc[1], geo, right, cells[:, -1, :], cells[:, -2, :])
    return faces


def _upwind_positions(upwind, x_f, x_c):
    """Normalised coordinates of the interior faces and their upwind cells.

    ``upwind`` holds the interior-face masks of ``v > 0`` and of its negation.
    """
    pos, neg = upwind
    x_f = x_f.reshape((1, -1, 1))
    x_c = x_c.reshape((1, -1, 1))
    x_d = x_f[:, 1:-1, :]
    x_C = (
        pos * x_c[:, :-1, :]
        + neg * x_c[:, 1:, :]
    )
    x_U = pos * np.concatenate(
        (x_f[:, 0:1, :], x_c[:, 0:-2, :]), axis=1
    ) + neg * np.concatenate(
        (x_c[:, 2:, :], x_f[:, -1:, :]), axis=1
    )
    x_D = (
        pos * x_c[:, 1:, :]
        + neg * x_c[:, :-1, :]
    )
    return (x_C - x_U) / (x_D - x_U), (x_d - x_U) / (x_D - x_U)


def _upwind_values(upwind, cells, faces):
    """Upwind-cell values, normalised values and downwind-upwind jumps at interior faces."""
    pos, neg = upwind
    c_C = (
        pos * cells[:, :-1, :]
        + neg * cells[:, 1:, :]
    )
    c_U = pos * np.concatenate(
        (faces[:, 0:1, :], cells[:, 0:-2, :]), axis=1
    ) + neg * np.concatenate(
        (cells[:, 2:, :], faces[:, -1:, :]), axis=1
    )
    c_D = (
        pos * cells[:, 1:, :]
        + neg * cells[:, :-1, :]
    )
    c_norm_C = np.zeros_like(c_C)
    dc_DU = c_D - c_U
    np.divide((c_C - c_U), dc_DU, out=c_norm_C, where=(dc_DU != 0))
    return c_C, c_norm_C, dc_DU


def interp_cntr_to_stagg_tvd(
    cell_centered_values, x_f, x_c=None, bc=None, v=0, tvd_limiter=None, axis=0
):
    """Perform TVD interpolation from cell centers to faces.

    A boundary given as ``{"outflow": True}`` gets the adjacent cell value at
    its face and no TVD correction there.

    Parameters
    ----------
    cell_centered_values : numpy.ndarray
        Cell-centered values.
    x_f : array_like
        Face coordinates along ``axis``.
    x_c : array_like, optional
        Cell-center coordinates. If omitted, midpoint locations are used.
    bc : tuple[dict | None, dict | None], optional
        Left and right boundary-condition dictionaries with keys ``a``, ``b``,
        and ``d``.
    v : float or array_like, optional
        Face velocity used to determine upwind/downwind directions.
    tvd_limiter : callable, optional
        Limiter function with signature ``phi(c_norm, x_norm_c, x_norm_d)``. If
        ``None``, the routine returns linear upwind interpolation without TVD
        correction.
    axis : int, optional
        Interpolation axis.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Interpolated staggered values and TVD correction term.
    """
    if bc is not None:
        bc, outflow = substitute_outflow_bc(bc, {"a": 1.0, "b": 0.0, "d": 0.0})
        if any(outflow):
            face, delta = interp_cntr_to_stagg_tvd(
                cell_centered_values, x_f, x_c, bc, v, tvd_limiter, axis
            )
            return _overwrite_outflow_faces(
                face, delta, cell_centered_values, outflow, axis)

    geo = _geometry_along(cell_centered_values.shape, axis)

    if x_c is None:
        x_c = 0.5 * (x_f[:-1] + x_f[1:])
    cells = cell_centered_values.reshape(geo.shape_t)
    if geo.shape_t[1] == 1:
        return _interp_single_cell(cells, x_f, x_c, bc, geo)

    faces = _wall_faces(cells, x_f, x_c, bc, geo)
    v = np.broadcast_to(np.asarray(v), geo.shape_f)
    pos = v.reshape(geo.shape_f_t)[:, 1:-1, :] > 0
    upwind = (pos, ~pos)
    x_norm_C, x_norm_d = _upwind_positions(upwind, x_f, x_c)
    c_C, c_norm_C, dc_DU = _upwind_values(upwind, cells, faces)
    faces = np.concatenate((faces[:, 0:1, :], c_C, faces[:, -1:, :]), axis=1)
    if tvd_limiter is None:
        return faces.reshape(geo.shape_f), np.zeros(geo.shape_f)
    delta = np.zeros(geo.shape_f_t)
    delta[:, 1:-1, :] = tvd_limiter(c_norm_C, x_norm_C, x_norm_d) * dc_DU
    faces += delta
    return faces.reshape(geo.shape_f), delta.reshape(geo.shape_f)


def create_staggered_array(array, shape, axis, x_f=None, x_c=None):
    """Create a face/staggered field from scalar, centered, or staggered input.

    Parameters
    ----------
    array : array_like
        Input values. May be scalar, centered, or already staggered.
    shape : tuple[int, ...] or int
        Target centered-field shape.
    axis : int
        Staggering axis.
    x_f, x_c : array_like, optional
        Face and center coordinates used when centered input must be
        interpolated to faces.

    Returns
    -------
    numpy.ndarray
        Broadcasted/interpolated array with staggered shape.
    """
    if not isinstance(shape, (list, tuple)):
        shape_f = [shape]
    else:
        shape_f = list(shape)
    if axis < 0:
        axis += len(shape)
    shape_f[axis] += 1
    shape_f = tuple(shape_f)

    array = np.asarray(array)
    if array.shape == shape_f:
        return array
    if array.size == 1:
        array = np.full(shape_f, array)
        return array

    array = _align_to_axes(array, shape, axis)
    if array.ndim != len(shape):
        raise ValueError("The array has the wrong number of dimensions.")
    if array.shape[axis] == shape[axis]:
        # interpolate to staggered positions
        array_f = interp_cntr_to_stagg(array, x_f, x_c, axis)
    else:
        array_f = array
    array_f = np.broadcast_to(array_f, shape_f)
    return array_f


def _align_to_axes(array, shape, axis):
    """Give a 1-D array the axes of ``shape``: along ``axis`` if its length is
    the cell or face count there, else along the last axis of that length."""
    if len(shape) == 1 or array.ndim != 1:
        return array
    shape_new = [1] * len(shape)
    if array.size in (shape[axis], shape[axis] + 1):
        shape_new[axis] = -1
    else:
        for i in range(len(shape) - 1, -1, -1):
            if array.size == shape[i]:
                shape_new[i] = shape[i]
                break
    return array.reshape(shape_new)


def _query_sides(bound_id):
    """Boundaries asked for: both when ``bound_id`` is ``None``, else the one given."""
    return [0, 1] if bound_id is None else [bound_id]


def _outflow_flags(bc, outflow, sides):
    """Outflow flag per queried side."""
    if isinstance(bc, dict) or len(outflow) == 1:
        return {sides[0]: outflow[0]}
    return {side: outflow[side] for side in sides}


def _overwrite_outflow_sides(result, cells, axis, sides, flags):
    """Replace outflow boundaries by the adjacent cell value and a zero gradient."""
    ax = axis % cells.ndim
    for k, side in enumerate(sides):
        if flags.get(side):
            adjacent = np.take(cells, [0 if side == 0 else -1], axis=ax)
            result[2 * k] = adjacent.reshape(np.shape(result[2 * k]))
            result[2 * k + 1] = np.zeros_like(np.asarray(result[2 * k + 1], dtype=float))
    return tuple(result)


def _boundary_values_single_cell(cells, system, geo, bounds):
    """Boundary values and gradients for a domain with a single cell."""
    boundary_values = [None, None]
    boundary_grads = [None, None]
    a, b, d, alpha_1, alpha_2, alpha_0, fctr = system

    for i in bounds:
        if i == 0:
            j = 1
            sgn = 1
        else:
            j = 0
            sgn = -1
        fctr_i = alpha_1 * (a[j] * (alpha_0[j] - alpha_2[i]) + b[j]) * fctr
        boundary_values[i] = _on_bc(a[i] * fctr_i, geo) * cells[:, 0, :]
        boundary_grads[i] = sgn * _on_bc(b[i] * fctr_i, geo) * cells[:, 0, :]

        fctr_m = (
            (a[j] * alpha_0[j] + b[j]) * d[i] - alpha_2[i] * a[i] * d[j]
        ) * fctr
        boundary_values[i][...] += _on_bc(fctr_m, geo)
        fctr_m = (
            (
                a[j] * (-alpha_0[i] * alpha_0[j] + alpha_2[i] * alpha_2[j])
                - alpha_0[i] * b[j]
            )
            * d[i]
            - alpha_2[i] * b[i] * d[j]
        ) * fctr
        boundary_grads[i][...] += sgn * _on_bc(fctr_m, geo)
        boundary_values[i] = boundary_values[i].reshape(geo.shape_bc)
        boundary_grads[i] = boundary_grads[i].reshape(geo.shape_bc)
    return boundary_values, boundary_grads


def _side_coords(x_f, x_c, i):
    """Coordinates of the wall, the nearest cell and the second cell at boundary ``i``."""
    if i == 0:
        return x_f[0], x_c[0], x_c[1]
    return x_f[-1], x_c[-1], x_c[-2]


def _side_alphas_of(walls, i):
    """Weights ``(alpha_1, alpha_2)`` of boundary ``i`` in the direction into the domain."""
    left, right = _wall_alphas(*walls)
    return left if i == 0 else (-right[0], -right[1])


def _boundary_values_two_cell_side(cells, bc_i, i, walls, geo):
    """Boundary value and gradient at boundary ``i`` from the two adjacent cells."""
    sgn, idx_0, idx_1 = (1, 0, 1) if i == 0 else (-1, -1, -2)
    x_wall, x_near, x_far = _side_coords(*walls, i)
    alpha_1, alpha_2 = _side_alphas_of(walls, i)
    a, b, d, alpha_0, fctr = _wall_factor(bc_i, geo, alpha_1, alpha_2, sgn)
    a_fctr = _on_bc(a * fctr, geo)
    b_fctr = _on_bc(b * fctr, geo)
    d_fctr = _on_bc(d * fctr, geo)
    c_near, c_far = cells[:, idx_0, :], cells[:, idx_1, :]
    value = d_fctr + a_fctr * (alpha_1 * c_near - alpha_2 * c_far)
    grad = -alpha_0 * d_fctr + b_fctr * (alpha_1 * c_near - alpha_2 * c_far)
    if np.any(fctr == 0.0):
        fltr = _on_bc(fctr == 0.0, geo)
        value[fltr] = (
            (x_far - x_wall) / (x_far - x_near)
        ) * c_near[fltr] + (
            (x_near - x_wall) / (x_near - x_far)
        ) * c_far[fltr]
        grad[fltr] = (1.0 / (x_far - x_near)) * (c_far[fltr] - c_near[fltr])
    return value.reshape(geo.shape_bc), grad.reshape(geo.shape_bc)


def _both_sides_bc(bc):
    """``bc`` as a pair of boundary dictionaries, ``None`` sides replaced by a zero condition."""
    if bc is None:
        return ({"a": 0, "b": 0, "d": 0}, {"a": 0, "b": 0, "d": 0})
    if not isinstance(bc, (tuple, list)) or len(bc) != 2:
        raise ValueError(
            "Boundary conditions must be a tuple of 2 dictionaries when 2 boundary conditions are required."
        )
    # a new tuple: the caller's bc is not changed
    return tuple({"a": 0, "b": 0, "d": 0} if side is None else side for side in bc)


def _normalize_boundary_bc(bc, bound_id, single_cell):
    """Return ``bc`` as a pair of boundary dictionaries (``None`` for an unused side)."""
    if bound_id is None or single_cell:
        return _both_sides_bc(bc)
    if bc is None:
        bc = {"a": 0, "b": 0, "d": 0}
    return (bc, None) if bound_id == 0 else (None, bc)


def compute_boundary_values(
    cell_centered_values, x_f, x_c=None, bc=None, axis=0, bound_id=None
):
    """Compute boundary values and boundary-normal gradients.

    Parameters
    ----------
    cell_centered_values : numpy.ndarray
        Cell-centered solution values.
    x_f : array_like
        Face coordinates along ``axis``.
    x_c : array_like, optional
        Cell-center coordinates.
    bc : dict or tuple[dict | None, dict | None], optional
        Boundary-condition data. For a single boundary query (``bound_id`` set),
        a single dictionary is accepted. For both boundaries, pass a
        two-element tuple. At a ``{"outflow": True}`` boundary the value is the
        adjacent cell value and the gradient is zero.
    axis : int, optional
        Axis normal to the boundary.
    bound_id : {0, 1} or None, optional
        Boundary selector. ``None`` returns both boundaries.

    Returns
    -------
    tuple
        If ``bound_id`` is ``None``:
        ``(value_left, grad_left, value_right, grad_right)``.
        Otherwise: ``(value, grad)`` for the requested boundary. Gradients are
        along the positive ``axis`` direction, NOT the outward normal used in
        the ``bc`` dictionaries (at the lower boundary they differ in sign).
    """
    if bc is not None:
        bc, outflow = substitute_outflow_bc(bc, {"a": 1.0, "b": 0.0, "d": 0.0})
        if any(outflow):
            result = list(compute_boundary_values(
                cell_centered_values, x_f, x_c, bc, axis, bound_id))
            sides = _query_sides(bound_id)
            return _overwrite_outflow_sides(
                result, cell_centered_values, axis, sides, _outflow_flags(bc, outflow, sides))

    geo = _geometry_along(cell_centered_values.shape, axis)
    cells = cell_centered_values.reshape(geo.shape_t)
    single_cell = geo.shape_t[1] == 1
    bounds = _query_sides(bound_id)

    bc = _normalize_boundary_bc(bc, bound_id, single_cell)

    boundary_values = [None, None]
    boundary_grads = [None, None]

    if single_cell:
        system = _single_cell_system(geo, x_f, x_c, bc)
        boundary_values, boundary_grads = _boundary_values_single_cell(cells, system, geo, bounds)
    else:
        if x_c is None:
            x_c = np.concatenate(
                (0.5 * (x_f[0:2] + x_f[1:3]), 0.5 * (x_f[-3:-1] + x_f[-2:]))
            )
        for i in bounds:
            boundary_values[i], boundary_grads[i] = _boundary_values_two_cell_side(
                cells, bc[i], i, (x_f, x_c), geo)

    if bound_id is None:
        return (
            boundary_values[0],
            boundary_grads[0],
            boundary_values[1],
            boundary_grads[1],
        )
    else:
        return boundary_values[bound_id], boundary_grads[bound_id]


def _boundary_value_matrix(values, i_f, i_c, geo, format):
    """Sparse matrix from cell values to boundary values in the requested format."""
    n0, n1, n2 = geo.shape_t
    shape_bc, shape_t = geo.shape_bc, geo.shape_t
    if format == "csc":
        matrix = csc_array(
            (values.ravel(), (i_f.ravel(), i_c.ravel())),
            shape=(math.prod(shape_bc), math.prod(shape_t)),
        )
    elif format == "csr":
        # Each boundary row has 2 entries (the two nearest cells).
        # Reorder to (n0, n2, 2) so entries are grouped by row.
        values_csr = values.transpose(0, 2, 1)
        cols_csr = i_c.transpose(0, 2, 1)
        num_rows = n0 * n2
        indptr = np.arange(0, 2 * num_rows + 1, 2, dtype=np.intp)
        matrix = csr_array(
            (values_csr.ravel(), cols_csr.ravel(), indptr),
            shape=(math.prod(shape_bc), math.prod(shape_t)),
        )
    else:
        raise ValueError(
            f"format must be 'csc' or 'csr', got {format!r}"
        )
    matrix.sort_indices()
    return matrix


def _boundary_value_indices(shape_t, idx_c_0, idx_c_1):
    """Cell, row and source-row indices of the two-point boundary value stencil."""
    n0, n1, n2 = shape_t
    i0 = np.arange(n0).reshape(-1, 1, 1)
    i2 = np.arange(n2).reshape(1, 1, -1)
    # Cell indices: two cells nearest the boundary
    i_c = np.ravel_multi_index(
        (i0, np.array([idx_c_0, idx_c_1]).reshape(1, -1, 1), i2), shape_t
    )
    # Row indices: both entries map to row 0 (single boundary row per batch)
    i_f = np.ravel_multi_index(
        (i0, np.array([0, 0]).reshape(1, -1, 1), i2), (n0, 1, n2)
    )
    i_f_bc = np.ravel_multi_index(
        (np.arange(n0).reshape(-1, 1), np.arange(n2).reshape(1, -1)), (n0, n2)
    ).reshape(n0, 1, n2)
    return i_c, i_f, i_f_bc


def _boundary_value_weights(bc, geo, walls, bound_id):
    """Weights of the two nearest cells and of the source in a boundary value."""
    sgn, idx_0, idx_1 = (1, 0, 1) if bound_id == 0 else (-1, -1, -2)
    x_wall, x_near, x_far = _side_coords(*walls, bound_id)
    n0, _, n2 = geo.shape_t
    values_bc = np.empty((n0, n2))
    values = np.empty((n0, 2, n2))

    alpha_1, alpha_2 = _side_alphas_of(walls, bound_id)
    a, _, d, _, fctr = _wall_factor(bc, geo, alpha_1, alpha_2, sgn)
    a_fctr = _on_bc(a * fctr, geo)
    d_fctr = _on_bc(d * fctr, geo)
    values[:, idx_0, :] = a_fctr * alpha_1
    values[:, idx_1, :] = -a_fctr * alpha_2
    values_bc[:, :] = d_fctr
    if np.any(fctr == 0.0):
        fltr = _on_bc(fctr == 0.0, geo)
        values[:, idx_0, :][fltr] = (x_far - x_wall) / (x_far - x_near)
        values[:, idx_1, :][fltr] = (x_near - x_wall) / (x_near - x_far)
        values_bc[fltr] = 0.0
    return values, values_bc


def _boundary_value_source(values_bc, i_f_bc, shape_d, geo):
    """Sparse matrix from the inhomogeneous boundary terms to the boundary values."""
    n_rows = math.prod(geo.shape_bc)
    if shape_d is None:
        return csc_array(
            (values_bc.ravel(), i_f_bc.ravel(), np.array([0, i_f_bc.size])),
            shape=(n_rows, 1),
        )
    num_cols = math.prod(shape_d)
    i_cols_bc = np.arange(num_cols, dtype=int).reshape(shape_d)
    i_cols_bc = np.broadcast_to(i_cols_bc, geo.shape_bc)
    return csc_array(
        (values_bc.ravel(), (i_f_bc.ravel(), i_cols_bc.ravel())),
        shape=(n_rows, num_cols),
    )


def construct_boundary_value_matrices(
    shape, x_f, x_c=None, bc=None, axis=0, bound_id=0, shape_d=None, format="csc"
):
    """Build matrices that evaluate boundary values from cell-centered unknowns.

    Parameters
    ----------
    shape : tuple[int, ...]
        Cell-centered field shape.
    x_f : array_like
        Face coordinates along ``axis``.
    x_c : array_like, optional
        Cell-center coordinates near the selected boundary.
    bc : dict, optional
        Boundary-condition dictionary with keys ``a``, ``b``, and ``d``.
    axis : int, optional
        Boundary-normal axis.
    bound_id : {0, 1}, optional
        ``0`` for the lower/left boundary, ``1`` for upper/right.
    shape_d : tuple[int, ...], optional
        Shape of external inhomogeneous source unknowns.
    format : {'csc', 'csr'}, optional
        Sparse format for the homogeneous matrix.

    Returns
    -------
    tuple
        ``(matrix, mat_bc)`` where ``matrix`` maps cell-centered values to
        boundary values and ``mat_bc`` maps inhomogeneous boundary terms.

    Raises
    ------
    ValueError
        If the axis has a single cell: the boundary value then depends on the
        conditions at both ends, which this function does not receive.
    NotImplementedError
        For an outflow boundary.
    """
    if is_outflow_bc(bc):
        raise NotImplementedError(
            "construct_boundary_value_matrices does not support an outflow boundary; "
            "the boundary value there is the adjacent cell value"
        )

    if bound_id not in (0, 1):
        raise ValueError("bound_id must be 0 or 1")

    # Trick: Reshape to triplet shape_t
    geo = _field_geometry(shape, axis)
    n1 = geo.shape_t[1]

    if n1 < 2:
        raise ValueError(
            "construct_boundary_value_matrices needs at least two cells along axis; "
            "with one cell the boundary value depends on both boundary conditions"
        )

    idx_c_0, idx_c_1 = (0, 1) if bound_id == 0 else (n1 - 2, n1 - 1)
    if x_c is None:
        if bound_id == 0:
            x_c = 0.5 * np.array([x_f[0] + x_f[1], x_f[1] + x_f[2]])
        else:
            x_c = 0.5 * np.array([x_f[-3] + x_f[-2], x_f[-2] + x_f[-1]])
    i_c, i_f, i_f_bc = _boundary_value_indices(geo.shape_t, idx_c_0, idx_c_1)
    values, values_bc = _boundary_value_weights(bc, geo, (x_f, x_c), bound_id)
    matrix = _boundary_value_matrix(values, i_f, i_c, geo, format)
    return matrix, _boundary_value_source(values_bc, i_f_bc, shape_d, geo)
