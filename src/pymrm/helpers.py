"""pymrm.helpers
=================

Utility helpers used throughout :mod:`pymrm`.

The functions in this module provide small building blocks that are reused in
multiple numerical routines.  They focus on preparing arrays for boundary
conditions and on constructing sparse coefficient matrices that are used in the
finite volume discretisation implemented by the package.

Functions
---------
``unwrap_bc_coeff``
    Expand boundary-condition coefficients to match an arbitrary domain shape.
``construct_coefficient_matrix``
    Create a sparse diagonal matrix from coefficient values.
``_sparse_array``
    Internal helper to construct a sparse array in the requested format.
"""

import math
from typing import NamedTuple
import numpy as np
from scipy.sparse import diags, csc_array, csr_array

try:  # private scipy kernel behind ``csc_array @ vector``
    from scipy.sparse._sparsetools import csc_matvec as _csc_matvec
except ImportError:  # pragma: no cover
    _csc_matvec = None


_FORMATS = {"csc", "csr"}


def _sparse_array(args, shape=None, format="csc"):
    """Create a sparse array in the requested format.

    This is an internal helper used throughout pymrm to honour the user's
    choice of sparse storage format while keeping the public API simple.

    Parameters
    ----------
    args : tuple
        Arguments forwarded to the sparse-array constructor.  Accepted forms
        are the same as for ``scipy.sparse.csc_array`` / ``csr_array``:
        ``(data, (rows, cols))``, ``(data, indices, indptr)``, etc.
    shape : tuple of int, optional
        Shape of the resulting sparse matrix.
    format : {'csc', 'csr'}, optional
        Sparse storage format.  Default is ``'csc'``.

    Returns
    -------
    scipy.sparse.csc_array or scipy.sparse.csr_array
    """
    if format not in _FORMATS:
        raise ValueError(f"format must be one of {_FORMATS!r}, got {format!r}")
    cls = csc_array if format == "csc" else csr_array
    if shape is not None:
        return cls(args, shape=shape)
    return cls(args)


def _csc_abs_matvec(jac, v):
    """``abs(jac) @ v`` without the checked copy that ``abs(jac)`` builds.

    For a canonical float64 ``csc_array`` ``jac`` and a float64 vector ``v`` of length
    ``jac.shape[1]`` (the caller checks both). scipy's ``@`` calls the same ``csc_matvec``
    kernel on the same absolute entries into a zeroed result, so the floats are identical
    (checked against scipy 1.18). Arrays that are inconsistent with their shape take the
    checked route, which raises as before.
    """
    n_cols = jac.shape[1]
    consistent = jac.indptr.size == n_cols + 1 and jac.indices.size == jac.data.size == jac.indptr[n_cols]
    if _csc_matvec is None or not consistent:  # pragma: no cover
        return abs(jac) @ v
    result = np.zeros(jac.shape[0])
    _csc_matvec(*jac.shape, jac.indptr, jac.indices, np.abs(jac.data), v, result)
    return result


def unwrap_bc_coeff(shape, bc_coeff, axis=0):
    """Expand boundary-condition coefficients to match a domain shape.

    Parameters
    ----------
    shape : tuple of int
        Target shape of the domain.
    bc_coeff : array_like
        Boundary-condition coefficient (e.g. ``a``, ``b`` or ``d`` terms).
    axis : int, optional
        Axis along which the coefficient applies.  The coefficient is expanded
        along this axis when needed.  Default is ``0``.

    Returns
    -------
    numpy.ndarray
        Array broadcast to ``shape`` containing the boundary-condition
        coefficients.
    """
    if not isinstance(shape, (list, tuple)):
        lgth_shape = 1
    else:
        lgth_shape = len(shape)

    a = np.array(bc_coeff)
    if a.ndim == (lgth_shape - 1):
        a = _insert_axis(a, axis)
    elif a.ndim != lgth_shape:
        shape_a = (1,) * (lgth_shape - a.ndim) + a.shape
        a = a.reshape(shape_a)
    return a


def _insert_axis(a, axis):
    """``np.expand_dims(a, axis)`` for one in-range integer axis, without its argument handling."""
    if isinstance(axis, (int, np.integer)) and -a.ndim - 1 <= axis <= a.ndim:
        pos = axis + a.ndim + 1 if axis < 0 else axis
        return a.reshape(a.shape[:pos] + (1,) + a.shape[pos:])
    return np.expand_dims(a, axis=axis)


def _bc_coeff(shape, bc_side, key, axis):
    """Coefficient ``key`` of one boundary condition expanded to ``shape`` (zeros if it is ``None``)."""
    if bc_side:
        return unwrap_bc_coeff(shape, bc_side[key], axis=axis)
    return np.zeros((1,) * len(shape))


def _unwrap_bc_side(shape, bc_side, axis):
    """Coefficients ``a``, ``b``, ``d`` of one boundary condition (zeros if it is ``None``)."""
    return [_bc_coeff(shape, bc_side, key, axis) for key in ["a", "b", "d"]]


def _wall_factor(bc_side, geo, alpha_1, alpha_2, sgn=1):
    """Coefficients of one boundary condition and its wall factor ``1 / (alpha_0 a + b)``.

    Returns ``(a, b, d, alpha_0, fctr)`` with ``alpha_0 = alpha_1 - alpha_2`` and ``a``
    multiplied by ``sgn``; ``fctr`` is left at zero where ``alpha_0 a + b`` vanishes.
    """
    a, b, d = _unwrap_bc_side(geo.shape, bc_side, geo.axis)
    alpha_0 = alpha_1 - alpha_2
    if sgn != 1:
        a *= sgn
    fctr = alpha_0 * a + b
    np.divide(1, fctr, out=fctr, where=(fctr != 0))
    return a, b, d, alpha_0, fctr


def _unwrap_bc_pair(shape, bc, axis):
    """Coefficients ``a``, ``b``, ``d`` of two boundary conditions, each a list over the sides."""
    return [[_bc_coeff(shape, bc_side, key, axis) for bc_side in bc] for key in ["a", "b", "d"]]


class _FieldGeometry(NamedTuple):
    """Shapes of a cell-centred field along one axis: triplet, face, boundary."""

    shape: tuple
    axis: int
    shape_t: tuple
    shape_f: tuple
    shape_f_t: tuple
    shape_bc: tuple
    shape_bc_d: tuple


def _field_geometry(shape, axis):
    """Triplet, face and boundary shapes of a field of ``shape`` along ``axis``.

    ``axis`` must be non-negative; the caller normalises it.
    """
    shape = tuple(shape)
    shape_t = (math.prod(shape[:axis]), shape[axis], math.prod(shape[axis + 1:]))
    shape_f = shape[:axis] + (shape[axis] + 1,) + shape[axis + 1:]
    return _FieldGeometry(
        shape, axis, shape_t, shape_f, (shape_t[0], shape_f[axis], shape_t[2]),
        shape[:axis] + (1,) + shape[axis + 1:], (shape_t[0], shape_t[2]))


class _BoundaryFaceTerms(NamedTuple):
    """Boundary-face coefficients with the index grids they are placed on."""

    values: np.ndarray
    values_bc: np.ndarray
    values_bc_src: np.ndarray
    i_c: np.ndarray
    i_f: np.ndarray
    i_f_bc: np.ndarray


def _boundary_face_indices(geo, cell_idx, face_idx):
    """Flat cell, face and boundary-face indices of the boundary stencils.

    ``cell_idx`` and ``face_idx`` list the positions along the axis of each stencil
    entry; the boundary faces are the first and the last face.
    """
    n0, _, n2 = geo.shape_t
    i0 = np.arange(n0).reshape(-1, 1, 1)
    i2 = np.arange(n2).reshape(1, 1, -1)

    def flat(idx, shape):
        return np.ravel_multi_index((i0, np.array(idx).reshape(1, -1, 1), i2), shape)

    return (flat(cell_idx, geo.shape_t), flat(face_idx, geo.shape_f_t),
            flat([0, geo.shape_f_t[1] - 1], geo.shape_f_t))


def _on_bc(value, geo):
    """``value`` broadcast over the boundary shape and flattened to ``(n0, n2)`` (a read-only view)."""
    if type(value) is np.ndarray and value.shape == geo.shape_bc:
        view = value.reshape(geo.shape_bc_d)
        view.flags.writeable = False     # as np.broadcast_to returns
        return view
    return np.broadcast_to(value, geo.shape_bc).reshape(geo.shape_bc_d)


def _wall_alphas(x_f, x_c):
    """Wall-stencil weights ``(alpha_1, alpha_2)`` of the left and the right boundary."""
    left = ((x_c[1] - x_f[0]) / ((x_c[0] - x_f[0]) * (x_c[1] - x_c[0])),
            (x_c[0] - x_f[0]) / ((x_c[1] - x_f[0]) * (x_c[1] - x_c[0])))
    right = (-(x_c[-2] - x_f[-1]) / ((x_c[-1] - x_f[-1]) * (x_c[-2] - x_c[-1])),
             -(x_c[-1] - x_f[-1]) / ((x_c[-2] - x_f[-1]) * (x_c[-2] - x_c[-1])))
    return left, right


class _SingleCellSystem(NamedTuple):
    """Boundary coefficients, wall weights and coupling factor of a one-cell domain."""

    a: list
    b: list
    d: list
    alpha_1: object
    alpha_2: list
    alpha_0: list
    fctr: np.ndarray


def _single_cell_system(geo, x_f, x_c, bc):
    """Weights and coupling factor of the two boundary conditions of a one-cell domain."""
    if x_c is None:
        x_c = 0.5 * (x_f[0:-1] + x_f[1:])
    a, b, d = _unwrap_bc_pair(geo.shape, bc, geo.axis)
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
    return _SingleCellSystem(a, b, d, alpha_1, alpha_2, alpha_0, fctr)


def _diagonal_sources(values_bc):
    """Sources per face and boundary when each boundary face depends on its own data only."""
    n0, _, n2 = values_bc.shape
    sources = np.zeros((n0, 2, 2, n2))
    sources[:, 0, 0, :] = values_bc[:, 0, :]
    sources[:, 1, 1, :] = values_bc[:, 1, :]
    return sources


def _bc_operators_by_boundary(terms, shapes_d, geo, format):
    """Operators and sources per boundary, for inhomogeneous terms on external vectors."""
    n0, n1, n2 = geo.shape_t
    n_faces = math.prod(geo.shape_f_t)
    values = terms.values.reshape((n0, -1, n2))
    bc_sources = []
    for i in range(2):
        shape_d = (1,) * len(geo.shape_bc) if shapes_d[i] is None else shapes_d[i]
        num_cols = math.prod(shape_d)
        i_cols_bc = np.arange(num_cols, dtype=int).reshape(shape_d)
        i_cols_bc = np.broadcast_to(i_cols_bc, geo.shape_bc).reshape(n0, 1, n2)
        faces = [0, 1] if n1 == 1 else [i]
        bc_sources.append(csc_array(
            (
                terms.values_bc_src[:, faces, i, :].ravel(),
                (
                    terms.i_f_bc[:, faces, :].ravel(),
                    np.broadcast_to(i_cols_bc, (n0, len(faces), n2)).ravel(),
                ),
            ),
            shape=(n_faces, num_cols),
        ))
    sides = (slice(0, 1), slice(-1, None)) if n1 == 1 else (slice(None, 2), slice(-2, None))
    matrices = [
        _sparse_array(
            (values[:, s, :].ravel(), (terms.i_f[:, s, :].ravel(), terms.i_c[:, s, :].ravel())),
            shape=(n_faces, math.prod(geo.shape_t)),
            format=format,
        )
        for s in sides
    ]
    return matrices[0], bc_sources[0], matrices[1], bc_sources[1]


def _stagger_shape(shape, axis):
    """``shape`` with the length along ``axis`` increased by one (face-centred length)."""
    return tuple(s if i != axis else s + 1 for i, s in enumerate(shape))


def _pairwise_coefficient_matrix(coefficients, shape, axis, fmt):
    """Coefficient matrix coupling two fields of shapes ``shape = (shape_rows, shape_cols)``."""
    shape_rows = shape[0]
    shape_cols = shape[1]
    working_shape = tuple(max(s1, s2) for s1, s2 in zip(shape_rows, shape_cols))
    if axis is not None:
        working_shape = _stagger_shape(working_shape, axis)
        if shape_rows[axis] + 1 == working_shape[axis]:
            shape_rows = _stagger_shape(shape_rows, axis)
        if shape_cols[axis] + 1 == working_shape[axis]:
            shape_cols = _stagger_shape(shape_cols, axis)
    if coefficients.shape == working_shape:
        coefficients_copy = coefficients
    else:
        coefficients_copy = np.array(coefficients)
        shape_coeff = (1,) * (
            len(working_shape) - coefficients_copy.ndim
        ) + coefficients_copy.shape
        coefficients_copy = coefficients_copy.reshape(shape_coeff)
        coefficients_copy = np.broadcast_to(coefficients_copy, working_shape)
    num_rows = math.prod(shape_rows)
    rows = np.arange(num_rows).reshape(shape_rows)
    rows = np.broadcast_to(rows, working_shape).ravel()
    num_cols = math.prod(shape_cols)
    cols = np.arange(num_cols).reshape(shape_cols)
    cols = np.broadcast_to(cols, working_shape).ravel()
    coeff_matrix = _sparse_array(
        (coefficients_copy.ravel(), (rows, cols)),
        shape=(num_rows, num_cols),
        format=fmt,
    )
    return coeff_matrix


def construct_coefficient_matrix(coefficients, shape=None, axis=None, format="csc"):
    """
    Build a sparse coefficient matrix with optional broadcasting and (row, col) coupling.

    Modes
    -----
    1. shape is None
       Treat coefficients as a flat sequence placed on the diagonal of an N×N matrix.

    2. shape is a single tuple, e.g. ``(Nz, Nr, ...)``
       Broadcast coefficients to that multidimensional shape (expanding leading size-1
       dimensions as needed) then place all values on the diagonal of a square matrix of
       size ``prod(shape)``. If ``axis`` is given, that dimension is first incremented
       by 1 (staggered / face-centred length) before broadcasting.

    3. shape is a pair of tuples: ``(shape_rows, shape_cols)``
       A dimension in either ``shape_rows`` or ``shape_cols`` can be singular. This can
       create a (possibly rectangular) matrix that couples two fields with different
       (but same-rank) shapes. The working broadcast shape is the element-wise maximum
       of ``shape_rows`` and ``shape_cols`` (adjusted by +1 along ``axis`` if provided;
       matching staggered dims in rows/cols are expanded too).

       Result::

           n_rows = prod(shape_rows)
           n_cols = prod(shape_cols)
           nnz    = prod(working_shape)

    Parameters
    ----------
    coefficients : array_like
        Scalar field to broadcast; shape must be broadcast-compatible with the target.
    shape : None | tuple | (tuple, tuple), optional
        Selects mode (see above).
    axis : int, optional
        Staggered axis: length along this axis is increased by 1 for broadcasting.
    format : {'csc', 'csr'}, optional
        Sparse storage format.  Default is ``'csc'``.

    Returns
    -------
    csc_array or csr_array
        Sparse matrix in the requested format.

    Notes
    -----
    - In mode (3) this is not a diagonal; it is a pointwise coupling pattern.
    - Broadcasting follows NumPy rules after auto-prepending leading 1s.

    Examples
    --------
    Diagonal from flat:
        A = construct_coefficient_matrix(np.ones(20))
    Diagonal from 2D field (staggered in axis 0):
        A = construct_coefficient_matrix(kz, shape=(Nz, Nr), axis=0)
    Rectangular coupling (cell centers -> axial faces):
        A = construct_coefficient_matrix(alpha, shape=((1, Nr), (Nz, Nr)), axis=0)
    """
    fmt = format
    if fmt not in ("csc", "csr"):
        raise ValueError(f"format must be one of {{'csc', 'csr'}}, got {fmt!r}")
    cls = csc_array if fmt == "csc" else csr_array
    if shape is None:
        coeff_matrix = cls(diags(coefficients.ravel(), format=fmt))
    elif all(isinstance(t, tuple) for t in shape):
        coeff_matrix = _pairwise_coefficient_matrix(coefficients, shape, axis, fmt)
    else:
        if axis is not None:
            shape = _stagger_shape(shape, axis)
        coefficients_copy = np.array(coefficients)
        shape_coeff = (1,) * (
            len(shape) - coefficients_copy.ndim
        ) + coefficients_copy.shape
        coefficients_copy = coefficients_copy.reshape(shape_coeff)
        coefficients_copy = np.broadcast_to(coefficients_copy, shape)
        coeff_matrix = cls(diags(coefficients_copy.ravel(), format=fmt))
    return coeff_matrix


def _format_coefficient(value):
    arr = np.asarray(value, dtype=float)
    if arr.size == 1:
        return f"{float(arr.ravel()[0]):.6g}"
    if arr.size <= 6:
        return "[" + ", ".join(f"{v:.6g}" for v in arr.ravel()) + "]"
    return f"[{arr.min():.6g} .. {arr.max():.6g}, shape {arr.shape}]"


def _classify_bc(a, b):
    a_zero = np.all(np.asarray(a, dtype=float) == 0.0)
    b_zero = np.all(np.asarray(b, dtype=float) == 0.0)
    a_nonzero = np.all(np.asarray(a, dtype=float) != 0.0)
    b_nonzero = np.all(np.asarray(b, dtype=float) != 0.0)
    if a_zero and b_nonzero:
        return "Dirichlet"
    if b_zero and a_nonzero:
        return "Neumann"
    if a_nonzero and b_nonzero:
        return "Robin"
    return "mixed or degenerate"


def describe_bc(bc, x_f=None, axis_name="x", var="c"):
    """Describe boundary-condition dictionaries as the equations they impose.

    pymrm boundary conditions read ``a * dc/dn + b * c = d`` with ``n`` the
    OUTWARD normal, so the same dictionary means opposite gradients at the two
    ends. This helper writes each condition out in terms of the axis direction,
    which makes sign errors visible.

    Parameters
    ----------
    bc : tuple[dict | None, dict | None]
        Lower and upper boundary dictionaries with keys ``a``, ``b``, ``d``.
    x_f : array_like, optional
        Face coordinates along the axis; used to print the boundary positions.
    axis_name : str, optional
        Name of the coordinate (default ``"x"``).
    var : str, optional
        Name of the field (default ``"c"``).

    Returns
    -------
    str
        One line per boundary, for example
        ``lower (x=0, outward normal -x): -1*dc/dx + 0*c = 2  [Neumann]``.
    """
    lines = []
    for side, sign, index in (("lower", "-", 0), ("upper", "+", -1)):
        entry = bc[index] if bc is not None else None
        where = f"{axis_name}={float(np.asarray(x_f)[index]):.6g}, " if x_f is not None else ""
        head = f"{side} ({where}outward normal {sign}{axis_name})"
        if entry is None:
            lines.append(f"{head}: None, treated as a = b = d = 0")
            continue
        if is_outflow_bc(entry):
            lines.append(f"{head}: outflow, face value = adjacent cell value, zero diffusive flux")
            continue
        a, b, d = (entry.get(key, 0.0) for key in ("a", "b", "d"))
        a_text = _format_coefficient(np.asarray(a, dtype=float) * (-1.0 if sign == "-" else 1.0) + 0.0)
        lines.append(
            f"{head}: {a_text}*d{var}/d{axis_name} + {_format_coefficient(b)}*{var} = "
            f"{_format_coefficient(d)}  [{_classify_bc(a, b)}]"
        )
    return "\n".join(lines)


def is_outflow_bc(bc_side):
    """Return True if a boundary dictionary is the pure-outflow marker.

    ``{"outflow": True}`` marks a boundary through which material leaves with
    the value of the adjacent cell (a stirred volume's exit, a tanks-in-series
    outlet). It cannot be expressed with ``a``, ``b`` and ``d``, which describe
    a reconstructed face value, so it is a separate marker.

    With convection into the face, the last cell is an implicit Euler step
    from its inflow face to the outlet: its value approximates the outlet
    value ``c(L)`` to second order, not the cell average. A higher-order (TVD)
    profile therefore drops to first order in a layer of 2 to 3 cells at the
    outlet (the error falls about fourfold per cell upstream); measure
    convergence order on the outlet value or at a fixed position.
    """
    if not isinstance(bc_side, dict) or not bc_side.get("outflow", False):
        return False
    if any(key in bc_side for key in ("a", "b", "d")):
        raise ValueError("an outflow boundary takes no 'a', 'b' or 'd' coefficients")
    return True


def substitute_outflow_bc(bc, replacement):
    """Replace outflow markers in ``bc`` by ``replacement``.

    Returns the new bc (tuple, or single dict) and the outflow flags.
    """
    if bc is None:
        return bc, (False, False)
    if isinstance(bc, dict):
        flag = is_outflow_bc(bc)
        return (dict(replacement) if flag else bc), (flag,)
    flags = tuple(is_outflow_bc(side) for side in bc)
    if not any(flags):
        return bc, flags
    replaced = [dict(replacement) if flag else side for side, flag in zip(bc, flags)]
    return (replaced if isinstance(bc, list) else tuple(replaced)), flags
