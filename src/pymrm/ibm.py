"""Directional ghost-cell immersed boundary method (IBM) for :mod:`pymrm`.

This module adds a *point-value*, Dirichlet directional ghost-cell immersed
boundary method for the non-equidistant finite-volume grids used throughout
:mod:`pymrm`.

Overview
--------
A signed-distance field (SDF) sampled at the cell centers flags every cell as
solid (``sdf < 0``) or fluid (``sdf >= 0``).  Wherever an axis-neighbor switches
region an *IBM point* (a wall crossing) is created.  For each IBM point the
value in the ghost cell is reconstructed by a second-order Lagrange
interpolation through the nodes ``{opposite neighbor, cell center, immersed-wall
value}`` -- exactly the construction used by
:func:`pymrm.operators.construct_grad_bc` for domain boundary conditions, but
now applied at an internal, off-grid wall location.

Applying the IBM to a sparse operator matrix folds each ghost column into the
owning row (adding to the cell center and the opposite neighbor, removing the
ghost entry) and produces an inhomogeneous *source* contribution whenever the
immersed-wall value is non-zero.  The source can be returned either as a vector
(wall values baked in) or as a matrix that maps a vector of per-IBM-point values
to the row source -- analogous to the ``shapes_d`` option of the gradient
operators, which is convenient for coupling and for Neumann-type conditions
imposed through probes.

The method is applied on *both* sides of the interface: the *outside* (fluid)
cells and the *inside* (solid) cells each receive their own IBM points and can
be given independent Dirichlet values.

Only nearest-neighbor stencils are supported and the operator matrices are
handled per row, so they are converted to (and assumed to be in) CSR format.

Sign convention
---------------
The returned modified matrix ``M`` and source ``g`` follow the same convention
as the gradient/divergence operators: the discretised quantity is
``M @ c + g`` (the source is *added*).  The optional per-row conditioning scale
is folded identically into the matrix and the source; use
:func:`apply_ibm_vector` to scale any independent right-hand-side term
consistently.
"""

from dataclasses import dataclass
import math
import warnings
import numpy as np
from scipy.sparse import csr_array, coo_array

from pymrm.grid import generate_grid

__all__ = ["IBM", "IBMSide", "construct_ibm", "apply_ibm", "apply_ibm_vector"]

# Clamp for the fractional wall distance theta = sdf_C / (sdf_C - sdf_ghost).
_THETA_MIN = 1e-4
_THETA_MAX = 1.0


@dataclass
class IBMSide:
    """Per-side IBM data for one region (``"outside"`` or ``"inside"``).

    Each entry (row of the arrays below) corresponds to one *ghost elimination*,
    i.e. the removal of a single solid/fluid ghost neighbor from the stencil of a
    cut cell along one axis and direction.  Each elimination *owns* exactly one
    IBM point (its own wall crossing); the point index therefore equals the
    elimination index, so ``values`` supplied to :func:`apply_ibm` has length
    :attr:`n_points`.

    Attributes
    ----------
    row : numpy.ndarray
        Flat (C-order) index of the cut cell that owns the row.
    ghost : numpy.ndarray
        Flat index of the eliminated ghost neighbor.
    opp : numpy.ndarray
        Flat index of the opposite neighbor, or ``-1`` when there is none
        (sandwich / domain-boundary cases).
    coef_c, coef_o : numpy.ndarray
        Lagrange coefficients of the ghost value on the cell center and the
        opposite neighbor.
    coef_w_self, coef_w_sib : numpy.ndarray
        Lagrange coefficients of the ghost value on the *own* wall crossing and,
        for the sandwich case, on the sibling wall crossing.
    sib : numpy.ndarray
        Index of the sibling elimination (the opposite-direction ghost of the
        same cell/axis) for sandwich points, else ``-1``.
    axis, direction : numpy.ndarray
        Axis and direction (``+1``/``-1``) of the eliminated ghost.
    crossing_key : numpy.ndarray
        Identifier shared by an outside point and its inside partner on the same
        face, so callers can align columns of the two source matrices.
    coords : numpy.ndarray
        Wall-crossing coordinates, shape ``(n_points, ndim)``.
    row_scale : numpy.ndarray
        Per-cell conditioning factor (length ``n_cells``; ``1`` off the cut
        cells).  Applied to the matrix, the source, and independent RHS terms.
    n_cells : int
        Total number of cells (``prod(shape)``).
    """

    row: np.ndarray
    ghost: np.ndarray
    opp: np.ndarray
    coef_c: np.ndarray
    coef_o: np.ndarray
    coef_w_self: np.ndarray
    coef_w_sib: np.ndarray
    sib: np.ndarray
    axis: np.ndarray
    direction: np.ndarray
    crossing_key: np.ndarray
    coords: np.ndarray
    row_scale: np.ndarray
    n_cells: int

    @property
    def n_points(self):
        """int: Number of IBM points (= ghost eliminations) on this side."""
        return self.row.size


@dataclass
class IBM:
    """Container holding the IBM data for both sides of the interface.

    Attributes
    ----------
    outside : IBMSide
        Data for the fluid (``sdf >= 0``) cut cells.
    inside : IBMSide
        Data for the solid (``sdf < 0``) cut cells.
    shape : tuple of int
        Cell-centered field shape the IBM was built for.
    """

    outside: IBMSide
    inside: IBMSide
    shape: tuple

    def side(self, name):
        """Return the :class:`IBMSide` for ``name`` (``"outside"``/``"inside"``)."""
        if name == "outside":
            return self.outside
        if name == "inside":
            return self.inside
        raise ValueError(f"side must be 'outside' or 'inside', got {name!r}")


def _neighbor(arr, axis, offset):
    """Shift ``arr`` so the result at cell ``i`` holds ``arr[i + offset]``.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to gather neighbors from.
    axis : int
        Axis along which to shift.
    offset : int
        Neighbor offset, ``+1`` or ``-1``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The neighbor-aligned values and a boolean mask that is ``False`` where
        the neighbor lies outside the domain.
    """
    n = arr.shape[axis]
    out = np.array(arr, copy=True)
    valid = np.ones(arr.shape, dtype=bool)
    dst = [slice(None)] * arr.ndim
    src = [slice(None)] * arr.ndim
    inv = [slice(None)] * arr.ndim
    if offset > 0:
        dst[axis] = slice(0, n - offset)
        src[axis] = slice(offset, n)
        inv[axis] = slice(n - offset, n)
    else:
        o = -offset
        dst[axis] = slice(o, n)
        src[axis] = slice(0, n - o)
        inv[axis] = slice(0, o)
    out[tuple(dst)] = arr[tuple(src)]
    valid[tuple(inv)] = False
    return out, valid


def _lagrange3(n0, n1, n2, p):
    """Evaluate the three quadratic Lagrange basis functions at ``p``.

    Parameters
    ----------
    n0, n1, n2 : numpy.ndarray
        Interpolation node coordinates.
    p : numpy.ndarray
        Evaluation coordinate.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Basis values ``(L0, L1, L2)`` such that the interpolant at ``p`` equals
        ``L0*f(n0) + L1*f(n1) + L2*f(n2)``.
    """
    l0 = (p - n1) * (p - n2) / ((n0 - n1) * (n0 - n2))
    l1 = (p - n0) * (p - n2) / ((n1 - n0) * (n1 - n2))
    l2 = (p - n0) * (p - n1) / ((n2 - n0) * (n2 - n1))
    return l0, l1, l2


def _build_side(sdf, shape, x_c, strides, region, other, rescale):
    """Build the :class:`IBMSide` for one region of the interface.

    Parameters
    ----------
    sdf : numpy.ndarray
        Cell-centered signed-distance field.
    shape : tuple of int
        Field shape.
    x_c : list of numpy.ndarray
        Cell-center coordinates per axis.
    strides : numpy.ndarray
        C-order flat strides of ``shape``.
    region : numpy.ndarray
        Boolean mask of the cells that own the rows on this side.
    other : numpy.ndarray
        Boolean mask of the opposite region (the ghost cells).
    rescale : bool
        Whether to compute a non-trivial per-row conditioning scale.

    Returns
    -------
    IBMSide
    """
    ndim = sdf.ndim
    n_cells = sdf.size
    flat_sdf = sdf.ravel()

    cols = {
        "row": [], "ghost": [], "opp": [],
        "coef_c": [], "coef_o": [], "coef_w_self": [], "coef_w_sib": [],
        "is_sw": [], "axis": [], "direction": [], "crossing_key": [], "coords": [],
    }

    warned = False
    for a in range(ndim):
        other_p, valid_p = _neighbor(other, a, +1)
        other_m, valid_m = _neighbor(other, a, -1)
        sdf_p, _ = _neighbor(sdf, a, +1)
        sdf_m, _ = _neighbor(sdf, a, -1)
        ghost_p = region & valid_p & other_p
        ghost_m = region & valid_m & other_m

        for direction in (+1, -1):
            gmask = ghost_p if direction == +1 else ghost_m
            omask = ghost_m if direction == +1 else ghost_p  # opposite is ghost?
            opp_valid = valid_m if direction == +1 else valid_p
            sdf_g_arr = sdf_p if direction == +1 else sdf_m
            sdf_o_arr = sdf_m if direction == +1 else sdf_p

            cells = np.flatnonzero(gmask.ravel())
            if cells.size == 0:
                continue
            multi = np.unravel_index(cells, shape)
            ia = multi[a]

            sdf_c = flat_sdf[cells]
            sdf_g = sdf_g_arr.ravel()[cells]
            theta = sdf_c / (sdf_c - sdf_g)
            theta = np.clip(theta, _THETA_MIN, _THETA_MAX)

            xc = x_c[a][ia]
            xg = x_c[a][ia + direction]
            xo = x_c[a][ia - direction]
            xw = xc + theta * (xg - xc)

            sandwich = omask.ravel()[cells]
            fallback = (~sandwich) & (~opp_valid.ravel()[cells])
            normal = (~sandwich) & (~fallback)

            coef_c = np.zeros(cells.size)
            coef_o = np.zeros(cells.size)
            coef_w_self = np.zeros(cells.size)
            coef_w_sib = np.zeros(cells.size)
            opp_lin = np.full(cells.size, -1, dtype=np.intp)

            # Normal case: quadratic through {opposite, center, wall}.
            if np.any(normal):
                l0, l1, l2 = _lagrange3(xo[normal], xc[normal], xw[normal], xg[normal])
                coef_o[normal] = l0
                coef_c[normal] = l1
                coef_w_self[normal] = l2
                opp_lin[normal] = cells[normal] - direction * strides[a]

            # Sandwich case: quadratic through {other wall, center, own wall}.
            if np.any(sandwich):
                sdf_o = sdf_o_arr.ravel()[cells][sandwich]
                theta_o = np.clip(
                    sdf_c[sandwich] / (sdf_c[sandwich] - sdf_o),
                    _THETA_MIN, _THETA_MAX,
                )
                xw_o = xc[sandwich] + theta_o * (xo[sandwich] - xc[sandwich])
                m0, m1, m2 = _lagrange3(xw_o, xc[sandwich], xw[sandwich], xg[sandwich])
                coef_w_sib[sandwich] = m0
                coef_c[sandwich] = m1
                coef_w_self[sandwich] = m2

            # Domain-boundary fallback: first-order 2-node (center, wall).
            if np.any(fallback):
                if not warned:
                    warnings.warn(
                        "IBM: immersed solid adjacent to the domain boundary; "
                        "falling back to first-order reconstruction for cells "
                        "without an in-domain opposite neighbor.",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    warned = True
                xcf = xc[fallback]
                xwf = xw[fallback]
                xgf = xg[fallback]
                coef_c[fallback] = (xgf - xwf) / (xcf - xwf)
                coef_w_self[fallback] = (xgf - xcf) / (xwf - xcf)

            ghost_lin = cells + direction * strides[a]

            # Canonical face key shared by outside/inside partners: the lower
            # cell of the pair along this axis, times ndim plus the axis.
            lower = np.minimum(cells, ghost_lin)
            key = lower * ndim + a

            coords = np.empty((cells.size, ndim))
            for j in range(ndim):
                coords[:, j] = x_c[j][multi[j]]
            coords[:, a] = xw

            cols["row"].append(cells.astype(np.intp))
            cols["ghost"].append(ghost_lin.astype(np.intp))
            cols["opp"].append(opp_lin)
            cols["coef_c"].append(coef_c)
            cols["coef_o"].append(coef_o)
            cols["coef_w_self"].append(coef_w_self)
            cols["coef_w_sib"].append(coef_w_sib)
            cols["is_sw"].append(sandwich)
            cols["axis"].append(np.full(cells.size, a, dtype=np.intp))
            cols["direction"].append(np.full(cells.size, direction, dtype=np.intp))
            cols["crossing_key"].append(key.astype(np.intp))
            cols["coords"].append(coords)

    if cols["row"]:
        row = np.concatenate(cols["row"])
        ghost = np.concatenate(cols["ghost"])
        opp = np.concatenate(cols["opp"])
        coef_c = np.concatenate(cols["coef_c"])
        coef_o = np.concatenate(cols["coef_o"])
        coef_w_self = np.concatenate(cols["coef_w_self"])
        coef_w_sib = np.concatenate(cols["coef_w_sib"])
        is_sw = np.concatenate(cols["is_sw"])
        axis = np.concatenate(cols["axis"])
        direction = np.concatenate(cols["direction"])
        crossing_key = np.concatenate(cols["crossing_key"])
        coords = np.concatenate(cols["coords"], axis=0)
    else:
        row = np.empty(0, dtype=np.intp)
        ghost = np.empty(0, dtype=np.intp)
        opp = np.empty(0, dtype=np.intp)
        coef_c = np.empty(0)
        coef_o = np.empty(0)
        coef_w_self = np.empty(0)
        coef_w_sib = np.empty(0)
        is_sw = np.empty(0, dtype=bool)
        axis = np.empty(0, dtype=np.intp)
        direction = np.empty(0, dtype=np.intp)
        crossing_key = np.empty(0, dtype=np.intp)
        coords = np.empty((0, ndim))

    # Link sibling eliminations for the sandwich case via (row, axis, direction).
    sib = np.full(row.size, -1, dtype=np.intp)
    if np.any(is_sw):
        lookup = {}
        for idx in range(row.size):
            lookup[(int(row[idx]), int(axis[idx]), int(direction[idx]))] = idx
        for idx in np.flatnonzero(is_sw):
            partner = lookup.get(
                (int(row[idx]), int(axis[idx]), -int(direction[idx])), -1
            )
            sib[idx] = partner

    # Geometric per-row conditioning scale.
    row_scale = np.ones(n_cells)
    if rescale and row.size:
        max_coef = np.maximum.reduce(
            [np.abs(coef_c), np.abs(coef_o), np.abs(coef_w_self), np.abs(coef_w_sib)]
        )
        per_cell = np.zeros(n_cells)
        np.maximum.at(per_cell, row, max_coef)
        cut = per_cell > 1.0
        row_scale[cut] = 1.0 / per_cell[cut]

    return IBMSide(
        row=row, ghost=ghost, opp=opp,
        coef_c=coef_c, coef_o=coef_o,
        coef_w_self=coef_w_self, coef_w_sib=coef_w_sib,
        sib=sib, axis=axis, direction=direction,
        crossing_key=crossing_key, coords=coords,
        row_scale=row_scale, n_cells=n_cells,
    )


def construct_ibm(sdf, x_f, x_c=None, rescale=True):
    """Build the immersed-boundary data from a cell-centered signed-distance field.

    Parameters
    ----------
    sdf : array_like
        Signed-distance field sampled at the cell centers.  Its shape defines the
        field shape.  Cells with ``sdf < 0`` are solid, ``sdf >= 0`` fluid.
    x_f : array_like or sequence of array_like
        Face coordinates.  For a 1-D field a single array (length
        ``shape[0] + 1``) is accepted; for an N-D field provide one array per
        axis.
    x_c : sequence of array_like, optional
        Cell-center coordinates per axis.  When omitted, arithmetic midpoints of
        ``x_f`` are used.
    rescale : bool, optional
        If ``True`` (default), compute a per-row conditioning scale that keeps
        the substitution coefficients of order one.  If ``False``, all row scales
        are one.

    Returns
    -------
    IBM
        Container with ``.outside`` and ``.inside`` :class:`IBMSide` data.

    Notes
    -----
    The reconstruction is second order (point-value Lagrange interpolation)
    everywhere except at immersed solids that touch the domain boundary, where a
    first-order fallback is used and a :class:`RuntimeWarning` is emitted.
    """
    sdf = np.asarray(sdf, dtype=float)
    shape = sdf.shape
    ndim = sdf.ndim

    if ndim == 1 and not (isinstance(x_f, (list, tuple)) and len(x_f) == 1):
        x_f = [x_f]
    if x_c is None:
        x_c = [None] * ndim
    elif ndim == 1 and not (isinstance(x_c, (list, tuple)) and len(x_c) == 1):
        x_c = [x_c]

    x_c_axes = []
    for a in range(ndim):
        _, xc = generate_grid(shape[a], x_f[a], generate_x_c=True, x_c=x_c[a])
        x_c_axes.append(np.asarray(xc, dtype=float))

    strides = np.array(
        [math.prod(shape[a + 1:]) for a in range(ndim)], dtype=np.intp
    )

    solid = sdf < 0.0
    fluid = ~solid

    outside = _build_side(sdf, shape, x_c_axes, strides, fluid, solid, rescale)
    inside = _build_side(sdf, shape, x_c_axes, strides, solid, fluid, rescale)
    return IBM(outside=outside, inside=inside, shape=tuple(shape))


def _read_entries(mat, rows, cols):
    """Return ``mat[rows[k], cols[k]]`` for every ``k`` as a dense 1-D array."""
    if rows.size == 0:
        return np.empty(0, dtype=float)
    values = np.asarray(mat[rows, cols])
    return values.ravel()


def _row_scaling_matrix(row_scale, n):
    """Return the sparse diagonal row-scaling operator as a ``csr_array``."""
    idx = np.arange(n, dtype=np.intp)
    return csr_array((row_scale, (idx, idx)), shape=(n, n))


def apply_ibm(mat, ibm, side="outside", values=None, return_bc="vector"):
    """Apply the immersed-boundary method to an operator matrix.

    The ghost columns of every cut-cell row are folded into that row: the cell
    center and the opposite neighbor absorb the reconstructed ghost value, the
    ghost entry is removed, and the immersed-wall value produces an
    inhomogeneous source contribution.  Cut-cell rows are optionally rescaled for
    conditioning (see :func:`construct_ibm`).

    Parameters
    ----------
    mat : scipy.sparse.sparray or spmatrix
        Operator matrix with a nearest-neighbor stencil.  It is converted to CSR
        (and assumed to be expressible in CSR).
    ibm : IBM
        Immersed-boundary data from :func:`construct_ibm`.
    side : {'outside', 'inside'}, optional
        Which region to apply the IBM on.
    values : array_like, optional
        Immersed-wall (Dirichlet) values, one per IBM point on ``side`` (length
        ``ibm.side(side).n_points``).  Only used when ``return_bc == 'vector'``.
        Defaults to zeros (homogeneous), giving a zero source vector.
    return_bc : {'vector', 'matrix'}, optional
        Form of the source contribution to return.  ``'vector'`` returns the
        source vector for the supplied ``values``.  ``'matrix'`` returns a sparse
        matrix ``G`` (shape ``(n_cells, n_points)``) such that the source for any
        value vector ``d`` is ``G @ d`` -- useful for coupling and probe-based
        Neumann conditions.

    Returns
    -------
    tuple
        ``(mat_mod, bc)`` where ``mat_mod`` is the modified CSR matrix and ``bc``
        is either the source vector (shape ``(n_cells,)``) or the source matrix.

    Notes
    -----
    The convention is ``value = mat_mod @ c + bc`` (the source is added), matching
    the gradient/divergence operators.  The per-row conditioning scale is folded
    into both ``mat_mod`` and ``bc``; scale any independent right-hand-side term
    with :func:`apply_ibm_vector` to stay consistent.
    """
    s = ibm.side(side)
    n = s.n_cells
    A = csr_array(mat)

    row, ghost, opp = s.row, s.ghost, s.opp
    v = _read_entries(A, row, ghost)

    # Matrix correction: fold ghost columns into center + opposite, zero ghost.
    has_opp = opp >= 0
    corr_rows = np.concatenate([row, row[has_opp], row])
    corr_cols = np.concatenate([row, opp[has_opp], ghost])
    corr_data = np.concatenate(
        [v * s.coef_c, v[has_opp] * s.coef_o[has_opp], -v]
    )
    correction = coo_array((corr_data, (corr_rows, corr_cols)), shape=(n, n))
    mat_mod = A + correction.tocsr()

    scaled = np.any(s.row_scale != 1.0)
    if scaled:
        scale = _row_scaling_matrix(s.row_scale, n)
        mat_mod = (scale @ mat_mod).tocsr()

    # Source matrix G (n_cells x n_points): each elimination contributes to its
    # own wall point and, for sandwich points, to the sibling wall point.
    npnt = s.n_points
    has_sib = s.sib >= 0
    own_idx = np.arange(npnt, dtype=np.intp)
    src_rows = np.concatenate([row, row[has_sib]])
    src_cols = np.concatenate([own_idx, s.sib[has_sib]])
    src_data = np.concatenate(
        [v * s.coef_w_self, v[has_sib] * s.coef_w_sib[has_sib]]
    )
    G = coo_array((src_data, (src_rows, src_cols)), shape=(n, npnt)).tocsr()
    if scaled:
        G = (_row_scaling_matrix(s.row_scale, n) @ G).tocsr()

    if return_bc == "matrix":
        return mat_mod, G
    if return_bc == "vector":
        if values is None:
            bc = np.zeros(n)
        else:
            values = np.asarray(values, dtype=float).ravel()
            if values.size != npnt:
                raise ValueError(
                    f"values must have length n_points={npnt}, got {values.size}"
                )
            bc = np.asarray(G @ values).ravel()
        return mat_mod, bc
    raise ValueError(f"return_bc must be 'vector' or 'matrix', got {return_bc!r}")


def apply_ibm_vector(vec, ibm, side="outside"):
    """Apply the IBM per-row conditioning scale to a vector.

    When the operator matrix is constant it is modified once with
    :func:`apply_ibm`; any independent right-hand-side / source vector must then
    be scaled with the same per-row factor to keep the linear system consistent.
    This helper performs that scaling.

    Parameters
    ----------
    vec : array_like
        Vector to scale.  Its leading dimension must be the number of cells; a
        flat array of length ``n_cells`` or an array with leading axis
        ``n_cells`` (e.g. ``(n_cells, k)``) is accepted.
    ibm : IBM
        Immersed-boundary data from :func:`construct_ibm`.
    side : {'outside', 'inside'}, optional
        Which region's row scale to apply.

    Returns
    -------
    numpy.ndarray
        The row-scaled vector, same shape as ``vec``.
    """
    s = ibm.side(side)
    vec = np.asarray(vec, dtype=float)
    scale = s.row_scale
    if vec.ndim > 1:
        scale = scale.reshape((-1,) + (1,) * (vec.ndim - 1))
    return scale * vec
