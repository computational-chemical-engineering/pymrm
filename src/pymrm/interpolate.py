"""Interpolation utilities between cell-centered and staggered grids."""

import math
import numpy as np
from scipy.sparse import csc_array, csr_array
from .helpers import unwrap_bc_coeff


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
    shape = list(cell_centered_values.shape)
    if axis < 0:
        axis += len(shape)
    shape_t = [
        math.prod(shape[:axis]),
        math.prod(shape[axis: axis + 1]),
        math.prod(shape[axis + 1:]),
    ]
    shape_f = shape.copy()
    shape_f[axis] += 1
    shape_f_t = shape_t.copy()
    shape_f_t[1] += 1
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


def interp_cntr_to_stagg_tvd(
    cell_centered_values, x_f, x_c=None, bc=None, v=0, tvd_limiter=None, axis=0
):
    """Perform TVD interpolation from cell centers to faces.

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
    shape = list(cell_centered_values.shape)
    if axis < 0:
        axis += len(shape)
    shape_t = [
        math.prod(shape[:axis]),
        math.prod(shape[axis: axis + 1]),
        math.prod(shape[axis + 1:]),
    ]  # reshape as a triplet
    shape_f = shape.copy()
    shape_f[axis] = shape[axis] + 1
    shape_f_t = shape_t.copy()
    shape_f_t[1] = shape_f[axis]
    shape_bc = shape_f.copy()
    shape_bc[axis] = 1
    shape_bc_d = [shape_t[0], shape_t[2]]

    if x_c is None:
        x_c = 0.5 * (x_f[:-1] + x_f[1:])
    cell_centered_values = cell_centered_values.reshape(shape_t)
    staggered_values = np.empty(shape_f_t)

    if shape_t[1] == 1:
        a, b, d = [
            [
                (
                    unwrap_bc_coeff(shape, bc_elem[key], axis=axis)
                    if bc_elem
                    else np.zeros((1,) * len(shape))
                )
                for bc_elem in bc
            ]
            for key in ["a", "b", "d"]
        ]
        alpha_1 = (x_f[1] - x_f[0]) / ((x_c[0] - x_f[0]) * (x_f[1] - x_c[0]))
        alpha_2_left = (x_c[0] - x_f[0]) / ((x_f[1] - x_f[0]) * (x_f[1] - x_c[0]))
        alpha_0_left = alpha_1 - alpha_2_left
        alpha_2_right = -(x_c[0] - x_f[1]) / ((x_f[0] - x_f[1]) * (x_f[0] - x_c[0]))
        alpha_0_right = alpha_1 - alpha_2_right
        fctr = (b[0] + alpha_0_left * a[0]) * (
            b[1] + alpha_0_right * a[1]
        ) - alpha_2_left * alpha_2_right * a[0] * a[1]
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        fctr_m = alpha_1 * a[0] * (a[1] * (alpha_0_right - alpha_2_left) + b[1]) * fctr
        fctr_m = fctr_m + np.zeros(shape_bc)
        fctr_m = np.reshape(fctr_m, shape_bc_d)
        staggered_values[:, 0, :] = fctr_m * cell_centered_values[:, 0, :]
        fctr_m = alpha_1 * a[1] * (a[0] * (alpha_0_left - alpha_2_right) + b[0]) * fctr
        fctr_m = fctr_m + np.zeros(shape_bc)
        fctr_m = np.reshape(fctr_m, shape_bc_d)
        staggered_values[:, 1, :] = fctr_m * cell_centered_values[:, 0, :]
        fctr_m = (
            (a[1] * alpha_0_right + b[1]) * d[0] - alpha_2_left * a[0] * d[1]
        ) * fctr
        fctr_m = fctr_m + np.zeros(shape_bc)
        fctr_m = np.reshape(fctr_m, shape_bc_d)
        staggered_values[:, 0, :] += fctr_m
        fctr_m = (
            (a[0] * alpha_0_left + b[0]) * d[1] - alpha_2_right * a[1] * d[0]
        ) * fctr
        fctr_m = fctr_m + np.zeros(shape_bc)
        fctr_m = np.reshape(fctr_m, shape_bc_d)
        staggered_values[:, 1, :] += fctr_m
        staggered_values.reshape(shape_f)
        delta_staggered_values = np.zeros(shape_f)
    else:
        # bc 0
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
        a_fctr = a_fctr + np.zeros(shape_bc)
        a_fctr = np.reshape(a_fctr, shape_bc_d)
        d_fctr = d * fctr
        d_fctr = d_fctr + np.zeros(shape_bc)
        d_fctr = np.reshape(d_fctr, shape_bc_d)
        staggered_values[:, 0, :] = d_fctr + a_fctr * (
            alpha_1 * cell_centered_values[:, 0, :]
            - alpha_2 * cell_centered_values[:, 1, :]
        )
        # bc 1
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
        a_fctr = a_fctr + np.zeros(shape_bc)
        a_fctr = np.reshape(a_fctr, shape_bc_d)
        d_fctr = d * fctr
        d_fctr = d_fctr + np.zeros(shape_bc)
        d_fctr = np.reshape(d_fctr, shape_bc_d)
        staggered_values[:, -1, :] = d_fctr + a_fctr * (
            alpha_1 * cell_centered_values[:, -1, :]
            - alpha_2 * cell_centered_values[:, -2, :]
        )

        v = np.broadcast_to(np.asarray(v), shape_f)
        v_t = v.reshape(shape_f_t)
        fltr_v_pos = v_t > 0

        x_f = x_f.reshape((1, -1, 1))
        x_c = x_c.reshape((1, -1, 1))
        x_d = x_f[:, 1:-1, :]
        x_C = (
            fltr_v_pos[:, 1:-1, :] * x_c[:, :-1, :]
            + ~fltr_v_pos[:, 1:-1, :] * x_c[:, 1:, :]
        )
        x_U = fltr_v_pos[:, 1:-1, :] * np.concatenate(
            (x_f[:, 0:1, :], x_c[:, 0:-2, :]), axis=1
        ) + ~fltr_v_pos[:, 1:-1, :] * np.concatenate(
            (x_c[:, 2:, :], x_f[:, -1:, :]), axis=1
        )
        x_D = (
            fltr_v_pos[:, 1:-1, :] * x_c[:, 1:, :]
            + ~fltr_v_pos[:, 1:-1, :] * x_c[:, :-1, :]
        )
        x_norm_C = (x_C - x_U) / (x_D - x_U)
        x_norm_d = (x_d - x_U) / (x_D - x_U)
        c_C = (
            fltr_v_pos[:, 1:-1, :] * cell_centered_values[:, :-1, :]
            + ~fltr_v_pos[:, 1:-1, :] * cell_centered_values[:, 1:, :]
        )
        c_U = fltr_v_pos[:, 1:-1, :] * np.concatenate(
            (staggered_values[:, 0:1, :], cell_centered_values[:, 0:-2, :]), axis=1
        ) + ~fltr_v_pos[:, 1:-1, :] * np.concatenate(
            (cell_centered_values[:, 2:, :], staggered_values[:, -1:, :]), axis=1
        )
        c_D = (
            fltr_v_pos[:, 1:-1, :] * cell_centered_values[:, 1:, :]
            + ~fltr_v_pos[:, 1:-1, :] * cell_centered_values[:, :-1, :]
        )
        c_norm_C = np.zeros_like(c_C)
        dc_DU = c_D - c_U
        np.divide((c_C - c_U), dc_DU, out=c_norm_C, where=(dc_DU != 0))
        staggered_values = np.concatenate(
            (staggered_values[:, 0:1, :], c_C, staggered_values[:, -1:, :]), axis=1
        )
        if tvd_limiter is None:
            delta_staggered_values = np.zeros(shape_f)
            staggered_values = staggered_values.reshape(shape_f)
        else:
            delta_staggered_values = np.zeros(shape_f_t)
            delta_staggered_values[:, 1:-1, :] = (
                tvd_limiter(c_norm_C, x_norm_C, x_norm_d) * dc_DU
            )
            staggered_values += delta_staggered_values
            delta_staggered_values = delta_staggered_values.reshape(shape_f)
            staggered_values = staggered_values.reshape(shape_f)
    return staggered_values, delta_staggered_values


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

    if len(shape) != 1 and array.ndim == 1:
        shape_new = [1] * len(shape)
        if array.size in (shape[axis], shape_f[axis]):
            shape_new[axis] = -1
        else:
            for i in range(len(shape) - 1, -1, -1):
                if array.size == shape[i]:
                    shape_new[i] = shape[i]
                    break
        array = array.reshape(shape_new)
    if array.ndim != len(shape):
        raise ValueError("The array has the wrong number of dimensions.")
    if array.shape[axis] == shape[axis]:
        # interpolate to staggered positions
        array_f = interp_cntr_to_stagg(array, x_f, x_c, axis)
    else:
        array_f = array
    array_f = np.broadcast_to(array_f, shape_f)
    return array_f


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
        two-element tuple.
    axis : int, optional
        Axis normal to the boundary.
    bound_id : {0, 1} or None, optional
        Boundary selector. ``None`` returns both boundaries.

    Returns
    -------
    tuple
        If ``bound_id`` is ``None``:
        ``(value_left, grad_left, value_right, grad_right)``.
        Otherwise: ``(value, grad)`` for the requested boundary.
    """
    shape = list(cell_centered_values.shape)
    if axis < 0:
        axis += len(shape)
    shape_t = [
        math.prod(shape[:axis]),
        math.prod(shape[axis: axis + 1]),
        math.prod(shape[axis + 1:]),
    ]  # reshape as a triplet
    shape_b_t = shape_t.copy()
    shape_b_t[1] = 2
    shape_bc = shape.copy()
    shape_bc[axis] = 1
    shape_bc_d = [shape_t[0], shape_t[2]]
    cell_centered_values = cell_centered_values.reshape(shape_t)

    if bound_id is None:
        bounds = [0, 1]
    else:
        bounds = [bound_id]

    if bound_id is None or shape_t[1] == 1:
        if bc is None:
            bc = ({"a": 0, "b": 0, "d": 0}, {"a": 0, "b": 0, "d": 0})
        elif not isinstance(bc, tuple) and len(bc) != 2:
            raise ValueError(
                "Boundary conditions must be a tuple of 2 dictionaries when 2 boundary conditions are required."
            )
        if bc[0] is None:
            bc[0] = {"a": 0, "b": 0, "d": 0}
        if bc[1] is None:
            bc[1] = {"a": 0, "b": 0, "d": 0}
    else:
        if bc is None:
            bc = {"a": 0, "b": 0, "d": 0}
        if bound_id == 0:
            bc = (bc, None)
        else:
            bc = (None, bc)

    boundary_values = [None, None]
    boundary_grads = [None, None]

    if shape_t[1] == 1:
        if x_c is None:
            x_c = 0.5 * (x_f[:-1] + x_f[1:])
        a, b, d = [
            [
                (
                    unwrap_bc_coeff(shape, bc_elem[key], axis=axis)
                    if bc_elem
                    else np.zeros((1,) * len(shape))
                )
                for bc_elem in bc
            ]
            for key in ["a", "b", "d"]
        ]
        alpha_1 = (x_f[1] - x_f[0]) / ((x_c[0] - x_f[0]) * (x_f[1] - x_c[0]))
        alpha_2 = [
            (x_c[0] - x_f[0]) / ((x_f[1] - x_f[0]) * (x_f[1] - x_c[0])),
            -(x_c[0] - x_f[1]) / ((x_f[0] - x_f[1]) * (x_f[0] - x_c[0])),
        ]
        alpha_0 = [alpha_1 - alpha_2[0], alpha_1 - alpha_2[1]]

        fctr = (b[0] + alpha_0[0] * a[0]) * (b[1] + alpha_0[1] * a[1]) - alpha_2[
            0
        ] * alpha_2[1] * a[0] * a[1]
        np.divide(1, fctr, out=fctr, where=(fctr != 0))

        for i in bounds:
            if i == 0:
                j = 1
                sgn = 1
            else:
                j = 0
                sgn = -1
            fctr_i = alpha_1 * (a[j] * (alpha_0[j] - alpha_2[i]) + b[j]) * fctr
            fctr_m = a[i] * fctr_i
            fctr_m = np.broadcast_to(fctr_m, shape_bc)
            fctr_m = np.reshape(fctr_m, shape_bc_d)
            boundary_values[i] = fctr_m * cell_centered_values[:, 0, :]
            fctr_m = b[i] * fctr_i
            fctr_m = np.broadcast_to(fctr_m, shape_bc)
            fctr_m = np.reshape(fctr_m, shape_bc_d)
            boundary_grads[i] = sgn * fctr_m * cell_centered_values[:, 0, :]

            fctr_m = (
                (a[j] * alpha_0[j] + b[j]) * d[i] - alpha_2[i] * a[i] * d[j]
            ) * fctr
            fctr_m = np.broadcast_to(fctr_m, shape_bc)
            fctr_m = np.reshape(fctr_m, shape_bc_d)
            boundary_values[i][...] += fctr_m
            fctr_m = (
                (
                    a[j] * (-alpha_0[i] * alpha_0[j] + alpha_2[i] * alpha_2[j])
                    - alpha_0[i] * b[j]
                )
                * d[i]
                - alpha_2[i] * b[i] * d[j]
            ) * fctr
            fctr_m = np.broadcast_to(fctr_m, shape_bc)
            fctr_m = np.reshape(fctr_m, shape_bc_d)
            boundary_grads[i][...] += sgn * fctr_m
            boundary_values[i] = boundary_values[i].reshape(shape_bc)
            boundary_grads[i] = boundary_grads[i].reshape(shape_bc)
    else:
        if x_c is None:
            x_c = np.concatenate(
                (0.5 * (x_f[0:2] + x_f[1:3]), 0.5 * (x_f[-3:-1] + x_f[-2:]))
            )
        for i in bounds:
            if i == 0:
                j = 1
                sgn = 1
                idx_0 = 0
                idx_1 = 1
            else:
                j = 0
                sgn = -1
                idx_0 = -1
                idx_1 = -2

            a, b, d = [
                (
                    unwrap_bc_coeff(shape, bc[i][key], axis=axis)
                    if bc[i]
                    else np.zeros((1,) * len(shape))
                )
                for key in ["a", "b", "d"]
            ]
            alpha_1 = (x_c[idx_1] - x_f[idx_0]) / (
                (x_c[idx_0] - x_f[idx_0]) * (x_c[idx_1] - x_c[idx_0])
            )
            alpha_2 = (x_c[idx_0] - x_f[idx_0]) / (
                (x_c[idx_1] - x_f[idx_0]) * (x_c[idx_1] - x_c[idx_0])
            )
            alpha_0 = alpha_1 - alpha_2
            a *= sgn
            fctr = alpha_0 * a + b
            np.divide(1, fctr, out=fctr, where=(fctr != 0))
            a_fctr = a * fctr
            a_fctr = np.broadcast_to(a_fctr, shape_bc)
            a_fctr = np.reshape(a_fctr, shape_bc_d)
            b_fctr = b * fctr
            b_fctr = np.broadcast_to(b_fctr, shape_bc)
            b_fctr = np.reshape(b_fctr, shape_bc_d)
            d_fctr = d * fctr
            d_fctr = np.broadcast_to(d_fctr, shape_bc)
            d_fctr = np.reshape(d_fctr, shape_bc_d)
            boundary_values[i] = d_fctr + a_fctr * (
                alpha_1 * cell_centered_values[:, idx_0, :]
                - alpha_2 * cell_centered_values[:, idx_1, :]
            )
            boundary_grads[i] = -alpha_0 * d_fctr + b_fctr * (
                alpha_1 * cell_centered_values[:, idx_0, :]
                - alpha_2 * cell_centered_values[:, idx_1, :]
            )
            if np.any(fctr == 0.0):
                fltr = np.reshape(np.broadcast_to((fctr == 0.0), shape_bc), shape_bc_d)
                boundary_values[i][fltr] = (
                    (x_c[idx_1] - x_f[idx_0]) / (x_c[idx_1] - x_c[idx_0])
                ) * cell_centered_values[:, idx_0, :][fltr] + (
                    (x_c[idx_0] - x_f[idx_0]) / (x_c[idx_0] - x_c[idx_1])
                ) * cell_centered_values[
                    :, idx_1, :
                ][
                    fltr
                ]
                boundary_grads[i][fltr] = (1.0 / (x_c[idx_1] - x_c[idx_0])) * (
                    cell_centered_values[:, idx_1, :][fltr]
                    - cell_centered_values[:, idx_0, :][fltr]
                )
            boundary_values[i] = boundary_values[i].reshape(shape_bc)
            boundary_grads[i] = boundary_grads[i].reshape(shape_bc)

    if bound_id is None:
        return (
            boundary_values[0],
            boundary_grads[0],
            boundary_values[1],
            boundary_grads[1],
        )
    else:
        return boundary_values[bound_id], boundary_grads[bound_id]


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
    """

    if bound_id not in (0, 1):
        raise ValueError("bound_id must be 0 or 1")

    # Trick: Reshape to triplet shape_t
    shape_t = (math.prod(shape[:axis]), shape[axis], math.prod(shape[axis + 1:]))
    shape_bc = shape[:axis] + (1,) + shape[axis + 1:]
    shape_bc_d = (shape_t[0], shape_t[2])

    # Handle special case with one cell in the dimension axis

    if bound_id == 0:
        idx_c_0 = 0
        idx_c_1 = 1
        idx_0 = 0
        idx_1 = 1
        sgn = 1
    else:
        idx_c_0 = shape_t[1] - 2
        idx_c_1 = shape_t[1] - 1
        idx_0 = -1
        idx_1 = -2
        sgn = -1
    if x_c is None:
        if bound_id == 0:
            x_c = 0.5 * np.array([x_f[0] + x_f[1], x_f[1] + x_f[2]])
        else:
            x_c = 0.5 * np.array([x_f[-3] + x_f[-2], x_f[-2] + x_f[-1]])
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
    values_bc = np.empty((shape_t[0], shape_t[2]))
    values = np.empty((shape_t[0], 2, shape_t[2]))

    # Get a, b, and d from dictionary
    a, b, d = [
        (
            unwrap_bc_coeff(shape, bc[key], axis=axis)
            if bc
            else np.zeros((1,) * len(shape))
        )
        for key in ["a", "b", "d"]
    ]
    a *= sgn
    alpha_1 = (x_c[idx_1] - x_f[idx_0]) / (
        (x_c[idx_0] - x_f[idx_0]) * (x_c[idx_1] - x_c[idx_0])
    )
    alpha_2 = (x_c[idx_0] - x_f[idx_0]) / (
        (x_c[idx_1] - x_f[idx_0]) * (x_c[idx_1] - x_c[idx_0])
    )
    alpha_0 = alpha_1 - alpha_2
    fctr = alpha_0 * a + b
    np.divide(1, fctr, out=fctr, where=(fctr != 0))
    a_fctr = a * fctr
    a_fctr = np.broadcast_to(a_fctr, shape_bc).reshape(shape_bc_d)
    d_fctr = d * fctr
    d_fctr = np.broadcast_to(d_fctr, shape_bc).reshape(shape_bc_d)
    values[:, idx_0, :] = a_fctr * alpha_1
    values[:, idx_1, :] = -a_fctr * alpha_2
    values_bc[:, :] = d_fctr
    if np.any(fctr == 0.0):
        fltr = np.reshape(np.broadcast_to((fctr == 0.0), shape_bc), shape_bc_d)
        values[:, idx_0, :][fltr] = (x_c[idx_1] - x_f[idx_0]) / (
            x_c[idx_1] - x_c[idx_0]
        )
        values[:, idx_1, :][fltr] = (x_c[idx_0] - x_f[idx_0]) / (
            x_c[idx_0] - x_c[idx_1]
        )
        values_bc[fltr] = 0.0

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
    if shape_d is None:
        mat_bc = csc_array(
            (values_bc.ravel(), i_f_bc.ravel(), np.array([0, i_f_bc.size])),
            shape=(math.prod(shape_bc), 1),
        )
    else:
        num_cols = math.prod(shape_d)
        i_cols_bc = np.arange(num_cols, dtype=int).reshape(shape_d)
        i_cols_bc = np.broadcast_to(i_cols_bc, shape_bc)
        mat_bc = csc_array(
            (values_bc.ravel(), (i_f_bc.ravel(), i_cols_bc.ravel())),
            shape=(math.prod(shape_bc), num_cols),
        )
    return matrix, mat_bc
