"""Directional ghost-cell immersed boundary method (IBM) for :mod:`pymrm`.

This module adds a *point-value*, Dirichlet directional ghost-cell immersed
boundary method for the non-equidistant finite-volume grids used throughout
:mod:`pymrm`.

Overview
--------
A signed-distance field (SDF) sampled at the **spatial** cell centres flags
every cell as solid (``sdf < 0``) or fluid (``sdf >= 0``).  Wherever an
axis-neighbour switches region a *wall crossing* is created.  Each crossing is
shared by exactly two rows of the operator matrix: the fluid (*outside*) cut
cell and the solid (*inside*) cut cell.  Both sides share the same geometric
wall position, so the IBM data is stored in a single :class:`IBM` object
indexed by crossing.

For each crossing the ghost value in the fluid row is reconstructed by a
second-order Lagrange interpolation through
``{opposite fluid neighbour, fluid cell centre, wall value}`` (and
symmetrically for the solid row).  This is the same Lagrange construction
used by :func:`pymrm.operators.construct_grad_bc` for domain boundary
conditions.

Multi-dimensional fields
------------------------
The field on which the IBM operator acts may have *non-spatial* axes
(components, phases, species, etc.) in addition to the spatial axes specified
by the ``axes`` argument to :func:`construct_ibm`.

The geometric Lagrange coefficients are identical for every *non-spatial
layer* at a given spatial crossing.  The modification of the operator matrix
therefore has a block structure: for crossing ``k`` and non-spatial layer
``j``, the entry :math:`v_{k,j} = A[r_{k,j}, g_{k,j}]` (matrix value at the
ghost column) may differ from layer to layer, and so may the supplied
Dirichlet wall values.  The full-field flat index decomposes as:

.. math::
    \\text{flat}(s, j) =
    \\underbrace{\\sum_i m_s[i] \\cdot d[\\text{axes}[i]]}_
        {\\text{spatial contrib.}}
    + \\underbrace{\\sum_i m_j[i] \\cdot d[\\text{ns-axes}[i]]}_
        {\\text{ns contrib.}}

where :math:`d[a]` is the C-order stride of axis ``a`` and the two
    contributions are **independent** (an outer sum over crossings x ns layers).
Sign convention
---------------
The modified matrix ``M`` and source ``g`` satisfy ``value = M @ c + g``
(source is *added*), matching the gradient/divergence operator convention.
The optional per-row conditioning scale is folded identically into the matrix
and the source.
"""

from dataclasses import dataclass
import math
import warnings
import numpy as np
from scipy.sparse import csr_array, coo_array

__all__ = ["IBM", "construct_ibm", "apply_ibm", "apply_ibm_vector"]

_THETA_MIN = 1e-4
_THETA_MAX = 1.0


# ---------------------------------------------------------------------------
# Internal per-side container (not part of the public API)
# ---------------------------------------------------------------------------

@dataclass
class _IBMSide:
    """Raw per-side data returned by :func:`_build_side` (internal use only)."""

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
    row_scale: np.ndarray   # shape (n_spatial_cells,)
    n_cells: int            # = n_spatial_cells

    @property
    def n_points(self):
        return self.row.size


# ---------------------------------------------------------------------------
# Public IBM container
# ---------------------------------------------------------------------------

@dataclass
class IBM:
    """Consolidated IBM crossing data for both sides of an immersed interface.

    Each entry (index ``k``) corresponds to one *wall crossing* -- a face
    between a fluid cell (outside) and a solid cell (inside).  The wall
    position ``coords[k]`` is shared by both sides.

    Indices are **spatial** flat indices (C-order in ``spatial_shape``), not
    full-field indices.  :func:`apply_ibm` expands them to the full field using
    the ``axes`` / ``shape`` information.

    Parameters indexed by crossing (length ``n_crossings``)
    --------------------------------------------------------
    n_crossings : int
        Number of wall crossings.
    coords : ndarray, shape (n_crossings, ndim_spatial)
        Physical coordinates of each wall crossing (spatial dimensions only).
    crossing_key : ndarray of int
        Canonical face identifier.
    axis : ndarray of int
        Spatial axis of each crossing.
    direction : ndarray of int
        Direction (+1 or -1) from ``row_out`` to ``ghost_out``.

    Outside (fluid cut-cell) fields
    --------------------------------
    row_out, ghost_out : ndarray of int
        Spatial flat indices of the fluid cut cell and its solid ghost
        neighbour.
    opp_out : ndarray of int
        Spatial flat index of the opposite fluid neighbour (``-1`` when
        unavailable).
    coef_c_out, coef_o_out, coef_w_out, coef_w_sib_out : ndarray of float
        Lagrange coefficients.
    sib_out : ndarray of int
        Crossing index of the sandwich sibling (``-1`` if not a sandwich).
    row_scale_out : ndarray of float, shape (n_spatial_cells,)
        Per-spatial-cell conditioning scale for the outside (fluid cut) rows.

    Inside (solid cut-cell) fields
    --------------------------------
    row_in, ghost_in, opp_in, coef_c_in, coef_o_in, coef_w_in,
    coef_w_sib_in, sib_in, row_scale_in : analogous inside fields.

    Shape information
    -----------------
    spatial_shape : tuple
        Shape of the SDF / spatial grid.
    shape : tuple
        Full field shape (spatial + non-spatial axes).
    axes : tuple of int
        Which axes of ``shape`` are spatial.
    ns_shape : tuple
        Non-spatial dimensions of ``shape``.
    ns_size : int
        Product of non-spatial dims (1 when purely spatial).
    n_cells : int
        Total number of cells, ``math.prod(shape)``.
    n_spatial_cells : int
        Number of spatial cells, ``math.prod(spatial_shape)``.
    """

    n_crossings: int
    coords: np.ndarray
    crossing_key: np.ndarray
    axis: np.ndarray
    direction: np.ndarray

    row_out: np.ndarray
    ghost_out: np.ndarray
    opp_out: np.ndarray
    coef_c_out: np.ndarray
    coef_o_out: np.ndarray
    coef_w_out: np.ndarray
    coef_w_sib_out: np.ndarray
    sib_out: np.ndarray
    row_scale_out: np.ndarray

    row_in: np.ndarray
    ghost_in: np.ndarray
    opp_in: np.ndarray
    coef_c_in: np.ndarray
    coef_o_in: np.ndarray
    coef_w_in: np.ndarray
    coef_w_sib_in: np.ndarray
    sib_in: np.ndarray
    row_scale_in: np.ndarray

    spatial_shape: tuple
    shape: tuple
    axes: tuple
    ns_shape: tuple
    ns_size: int
    n_cells: int
    n_spatial_cells: int


# ---------------------------------------------------------------------------
# Geometry helpers (spatial ↔ full-field index mapping)
# ---------------------------------------------------------------------------

def _spatial_contributions(spatial_flat, shape, axes):
    """Contribution of spatial multi-indices to the C-order full flat index.

    For each spatial flat index ``s``, this returns:

        ``sum(multi_s[i] * full_stride[axes[i]] for i in range(len(axes)))``

    Parameters
    ----------
    spatial_flat : array_like of int
        Flat indices in the spatial subspace.
    shape : tuple
        Full field shape.
    axes : tuple of int
        Which axes of ``shape`` are spatial.

    Returns
    -------
    numpy.ndarray of int
        Same shape as ``spatial_flat``.
    """
    spatial_flat = np.asarray(spatial_flat, dtype=np.intp)
    spatial_shape = tuple(shape[a] for a in axes)
    full_strides = [math.prod(shape[a + 1:]) for a in range(len(shape))]

    multi = np.unravel_index(spatial_flat, spatial_shape)
    contrib = np.zeros(spatial_flat.shape, dtype=np.intp)
    for i, a in enumerate(axes):
        contrib += multi[i] * full_strides[a]
    return contrib


def _ns_contributions(shape, axes):
    """Contribution of non-spatial multi-indices to the C-order full flat index.

    Returns a 1-D array of shape ``(ns_size,)`` where
    ``ns_size = prod(shape[a] for a not in axes)``.

    Parameters
    ----------
    shape : tuple
        Full field shape.
    axes : tuple of int
        Spatial axes.

    Returns
    -------
    numpy.ndarray of int, shape (ns_size,)
    """
    axes_set = set(axes)
    ns_axes = [a for a in range(len(shape)) if a not in axes_set]
    ns_shape = tuple(shape[a] for a in ns_axes)
    ns_size = math.prod(ns_shape) if ns_shape else 1

    if ns_size == 1:
        return np.zeros(1, dtype=np.intp)

    full_strides = [math.prod(shape[a + 1:]) for a in range(len(shape))]
    ns_multi = np.unravel_index(np.arange(ns_size, dtype=np.intp), ns_shape)
    contrib = np.zeros(ns_size, dtype=np.intp)
    for i, a in enumerate(ns_axes):
        contrib += ns_multi[i] * full_strides[a]
    return contrib


def _expand_row_scale(scale_spatial, ibm):
    """Expand a per-spatial-cell scale to every cell in the full field.

    Parameters
    ----------
    scale_spatial : ndarray, shape (n_spatial_cells,)
    ibm : IBM

    Returns
    -------
    ndarray, shape (n_cells,)
    """
    ns = ibm.ns_size
    ns_c = _ns_contributions(ibm.shape, ibm.axes)
    sc_all = _spatial_contributions(
        np.arange(ibm.n_spatial_cells, dtype=np.intp), ibm.shape, ibm.axes
    )
    full_flat = (sc_all[:, np.newaxis] + ns_c[np.newaxis, :]).ravel()
    full_scale = np.ones(ibm.n_cells)
    full_scale[full_flat] = np.repeat(scale_spatial, ns)
    return full_scale


# ---------------------------------------------------------------------------
# Internal geometry construction
# ---------------------------------------------------------------------------

def _neighbor(arr, axis, offset):
    """Return the axis-shifted copy of *arr* and a boolean validity mask."""
    n = arr.shape[axis]
    out = arr.copy()
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
    """Quadratic Lagrange basis values at *p* over nodes (*n0*, *n1*, *n2*)."""
    l0 = (p - n1) * (p - n2) / ((n0 - n1) * (n0 - n2))
    l1 = (p - n0) * (p - n2) / ((n1 - n0) * (n1 - n2))
    l2 = (p - n0) * (p - n1) / ((n2 - n0) * (n2 - n1))
    return l0, l1, l2


def _build_side(sdf, spatial_shape, x_c, strides, region, other, rescale):
    """Build an :class:`_IBMSide` for one region of the immersed interface.

    Parameters
    ----------
    sdf : ndarray, shape = spatial_shape
        Cell-centred signed-distance field.
    spatial_shape : tuple
        Spatial grid shape.
    x_c : list of ndarray
        Cell-centre coordinates, one 1-D array per spatial axis.
    strides : ndarray of int
        C-order flat strides of ``spatial_shape``.
    region : ndarray of bool
        Cells that own rows on this side.
    other : ndarray of bool
        Cells in the opposite region (ghost cells).
    rescale : bool
        Whether to compute per-row conditioning scale.
    """
    ndim_s = sdf.ndim
    n_spatial = sdf.size
    flat_sdf = sdf.ravel()

    cols = {k: [] for k in (
        "row", "ghost", "opp", "coef_c", "coef_o",
        "coef_w_self", "coef_w_sib", "is_sw",
        "axis", "direction", "crossing_key", "coords",
    )}

    warned = False
    for a in range(ndim_s):
        other_p, valid_p = _neighbor(other, a, +1)
        other_m, valid_m = _neighbor(other, a, -1)
        sdf_p, _ = _neighbor(sdf, a, +1)
        sdf_m, _ = _neighbor(sdf, a, -1)

        for direction in (+1, -1):
            gmask = (region & valid_p & other_p) if direction == +1 else (region & valid_m & other_m)
            omask = (region & valid_m & other_m) if direction == +1 else (region & valid_p & other_p)
            opp_valid = valid_m if direction == +1 else valid_p
            sdf_g_arr = sdf_p if direction == +1 else sdf_m
            sdf_o_arr = sdf_m if direction == +1 else sdf_p

            cells = np.flatnonzero(gmask.ravel())
            if cells.size == 0:
                continue
            multi = np.unravel_index(cells, spatial_shape)
            ia = multi[a]

            sdf_c = flat_sdf[cells]
            sdf_g = sdf_g_arr.ravel()[cells]
            theta = np.clip(sdf_c / (sdf_c - sdf_g), _THETA_MIN, _THETA_MAX)

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

            if np.any(normal):
                l0, l1, l2 = _lagrange3(xo[normal], xc[normal], xw[normal], xg[normal])
                coef_o[normal] = l0
                coef_c[normal] = l1
                coef_w_self[normal] = l2
                opp_lin[normal] = cells[normal] - direction * strides[a]

            if np.any(sandwich):
                sdf_o = sdf_o_arr.ravel()[cells][sandwich]
                theta_o = np.clip(
                    sdf_c[sandwich] / (sdf_c[sandwich] - sdf_o), _THETA_MIN, _THETA_MAX
                )
                xw_o = xc[sandwich] + theta_o * (xo[sandwich] - xc[sandwich])
                m0, m1, m2 = _lagrange3(xw_o, xc[sandwich], xw[sandwich], xg[sandwich])
                coef_w_sib[sandwich] = m0
                coef_c[sandwich] = m1
                coef_w_self[sandwich] = m2

            if np.any(fallback):
                if not warned:
                    warnings.warn(
                        "IBM: immersed solid adjacent to the domain boundary; "
                        "falling back to first-order reconstruction.",
                        RuntimeWarning, stacklevel=4,
                    )
                    warned = True
                xcf, xwf, xgf = xc[fallback], xw[fallback], xg[fallback]
                coef_c[fallback] = (xgf - xwf) / (xcf - xwf)
                coef_w_self[fallback] = (xgf - xcf) / (xwf - xcf)

            ghost_lin = cells + direction * strides[a]
            lower = np.minimum(cells, ghost_lin)
            key = lower * ndim_s + a

            coords = np.empty((cells.size, ndim_s))
            for j in range(ndim_s):
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
        direction_arr = np.concatenate(cols["direction"])
        crossing_key = np.concatenate(cols["crossing_key"])
        coords = np.concatenate(cols["coords"], axis=0)
    else:
        row = ghost = opp = np.empty(0, dtype=np.intp)
        coef_c = coef_o = coef_w_self = coef_w_sib = np.empty(0)
        is_sw = np.empty(0, dtype=bool)
        axis = direction_arr = crossing_key = np.empty(0, dtype=np.intp)
        coords = np.empty((0, ndim_s))

    # Link sandwich siblings within this side.
    sib = np.full(row.size, -1, dtype=np.intp)
    if np.any(is_sw):
        lookup = {}
        for idx in range(row.size):
            lookup[(int(row[idx]), int(axis[idx]), int(direction_arr[idx]))] = idx
        for idx in np.flatnonzero(is_sw):
            partner = lookup.get(
                (int(row[idx]), int(axis[idx]), -int(direction_arr[idx])), -1
            )
            sib[idx] = partner

    # Per-row conditioning scale (per spatial cell).
    row_scale = np.ones(n_spatial)
    if rescale and row.size:
        max_coef = np.maximum.reduce(
            [np.abs(coef_c), np.abs(coef_o), np.abs(coef_w_self), np.abs(coef_w_sib)]
        )
        per_cell = np.zeros(n_spatial)
        np.maximum.at(per_cell, row, max_coef)
        cut = per_cell > 1.0
        row_scale[cut] = 1.0 / per_cell[cut]

    return _IBMSide(
        row=row, ghost=ghost, opp=opp,
        coef_c=coef_c, coef_o=coef_o,
        coef_w_self=coef_w_self, coef_w_sib=coef_w_sib,
        sib=sib, axis=axis, direction=direction_arr,
        crossing_key=crossing_key, coords=coords,
        row_scale=row_scale, n_cells=n_spatial,
    )


def _remap_sib(sib_old, sort_idx):
    """Remap sandwich sibling indices from unsorted to sorted crossing order."""
    n = len(sib_old)
    if n == 0:
        return sib_old.copy()
    old_to_new = np.empty(n, dtype=np.intp)
    old_to_new[sort_idx] = np.arange(n, dtype=np.intp)
    sib_sorted = sib_old[sort_idx]
    valid = sib_sorted >= 0
    sib_new = np.full(n, -1, dtype=np.intp)
    sib_new[valid] = old_to_new[sib_sorted[valid]]
    return sib_new


def _row_scaling_matrix(full_scale, n):
    """Sparse diagonal row-scaling operator as a ``csr_array``."""
    idx = np.arange(n, dtype=np.intp)
    return csr_array((full_scale, (idx, idx)), shape=(n, n))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def construct_ibm(sdf, x_c, axes=None, shape=None, rescale=True):
    """Build the immersed-boundary data from a spatial signed-distance field.

    Parameters
    ----------
    sdf : array_like
        Signed-distance field sampled at the **spatial** cell centres.
        ``sdf.shape`` must equal ``tuple(shape[a] for a in axes)``.
        Cells with ``sdf < 0`` are solid; ``sdf >= 0`` is fluid.
    x_c : array_like or list of array_like
        Cell-centre coordinates.  For a 1-D spatial grid a single 1-D array
        is accepted; for N-D spatial grids supply a list of N 1-D arrays (one
        per spatial axis).  The i-th element has length ``sdf.shape[i]``.
    axes : tuple of int, optional
        Which axes of the **full field** array correspond to spatial
        coordinates.  Length must equal ``sdf.ndim``.  Defaults to
        ``tuple(range(sdf.ndim))`` (all axes are spatial).
    shape : tuple of int, optional
        Full field shape, including any non-spatial dimensions (components,
        phases, species, etc.).  Defaults to ``sdf.shape`` (purely spatial,
        no non-spatial axes).
    rescale : bool, optional
        If ``True`` (default), apply a geometric per-row conditioning scale.

    Returns
    -------
    IBM
        Container holding per-crossing geometry, Lagrange coefficients, and
        per-row conditioning scales for both sides of the interface.

    Notes
    -----
    The IBM uses cell-centre coordinates directly.  Face coordinates are not
    required because the Lagrange interpolation nodes are the cell centres and
    the wall position is found from the SDF: ``x_w = x_c + θ (x_ghost − x_c)``.

    Every wall crossing ``k`` is the face between one fluid cell and one solid
    cell.  ``ibm.coords[k]``, ``ibm.row_out[k]`` (fluid side, spatial flat
    index), and ``ibm.row_in[k]`` (solid side, spatial flat index) all refer
    to the same physical wall crossing.
    """
    sdf = np.asarray(sdf, dtype=float)
    ndim_s = sdf.ndim
    spatial_shape = sdf.shape

    # Normalise x_c to a list of 1-D arrays.
    if isinstance(x_c, np.ndarray) and x_c.ndim == 1:
        x_c = [x_c]
    else:
        x_c = [np.asarray(xci, dtype=float) for xci in x_c]
    if len(x_c) != ndim_s:
        raise ValueError(
            f"len(x_c)={len(x_c)} must equal sdf.ndim={ndim_s}"
        )

    # Normalise axes and shape.
    if axes is None and shape is None:
        axes = tuple(range(ndim_s))
        shape = spatial_shape
    elif shape is None:
        axes = tuple(int(a) for a in axes)
        shape = spatial_shape
    else:
        axes = tuple(int(a) for a in axes)
        shape = tuple(shape)
        if tuple(shape[a] for a in axes) != spatial_shape:
            raise ValueError(
                f"sdf.shape={spatial_shape} is inconsistent with "
                f"shape={shape} at axes={axes}"
            )

    axes_set = set(axes)
    ns_axes = [a for a in range(len(shape)) if a not in axes_set]
    ns_shape = tuple(shape[a] for a in ns_axes)
    ns_size = math.prod(ns_shape) if ns_shape else 1
    n_cells = math.prod(shape)
    n_spatial_cells = math.prod(spatial_shape)

    strides = np.array(
        [math.prod(spatial_shape[a + 1:]) for a in range(ndim_s)], dtype=np.intp
    )

    solid = sdf < 0.0
    fluid = ~solid

    out_s = _build_side(sdf, spatial_shape, x_c, strides, fluid, solid, rescale)
    in_s = _build_side(sdf, spatial_shape, x_c, strides, solid, fluid, rescale)

    # Pair outside and inside crossings by sorting on the canonical face key.
    sort_out = np.argsort(out_s.crossing_key, kind="stable")
    sort_in = np.argsort(in_s.crossing_key, kind="stable")

    keys_out = out_s.crossing_key[sort_out]
    keys_in = in_s.crossing_key[sort_in]

    if keys_out.size != keys_in.size or not np.array_equal(keys_out, keys_in):
        raise RuntimeError(
            "IBM: outside and inside crossings do not pair up "
            "(crossing key mismatch). This is a bug — please report it."
        )

    return IBM(
        n_crossings=out_s.n_points,
        coords=out_s.coords[sort_out],
        crossing_key=keys_out,
        axis=out_s.axis[sort_out],
        direction=out_s.direction[sort_out],

        row_out=out_s.row[sort_out],
        ghost_out=out_s.ghost[sort_out],
        opp_out=out_s.opp[sort_out],
        coef_c_out=out_s.coef_c[sort_out],
        coef_o_out=out_s.coef_o[sort_out],
        coef_w_out=out_s.coef_w_self[sort_out],
        coef_w_sib_out=out_s.coef_w_sib[sort_out],
        sib_out=_remap_sib(out_s.sib, sort_out),
        row_scale_out=out_s.row_scale,

        row_in=in_s.row[sort_in],
        ghost_in=in_s.ghost[sort_in],
        opp_in=in_s.opp[sort_in],
        coef_c_in=in_s.coef_c[sort_in],
        coef_o_in=in_s.coef_o[sort_in],
        coef_w_in=in_s.coef_w_self[sort_in],
        coef_w_sib_in=in_s.coef_w_sib[sort_in],
        sib_in=_remap_sib(in_s.sib, sort_in),
        row_scale_in=in_s.row_scale,

        spatial_shape=tuple(spatial_shape),
        shape=shape,
        axes=axes,
        ns_shape=ns_shape,
        ns_size=ns_size,
        n_cells=n_cells,
        n_spatial_cells=n_spatial_cells,
    )


# ---------------------------------------------------------------------------
# Application helpers
# ---------------------------------------------------------------------------

def _read_entries(mat, rows, cols):
    """Return ``mat[rows[k], cols[k]]`` for every ``k`` as a 1-D array."""
    if rows.size == 0:
        return np.empty(0, dtype=float)
    return np.asarray(mat[rows, cols]).ravel()


def _normalize_values(values_outside, values_inside, npnt, ns):
    """Normalise IBM wall values to shape ``(npnt, ns)``.

    Rules
    -----
    * Both ``None`` → zeros.
    * One ``None`` → same as the other.
    * Scalar or 1-D array of length ``npnt`` → broadcast over ns.
    * Shape ``(npnt, ns)`` → used as-is.
    """
    def _coerce(v, name):
        if v is None:
            return None
        v = np.asarray(v, dtype=float)
        if v.ndim == 0:
            return np.full((npnt, ns), float(v))
        if v.ndim == 1:
            if v.size != npnt:
                raise ValueError(
                    f"{name} length {v.size} != n_crossings {npnt}"
                )
            return np.broadcast_to(v[:, np.newaxis], (npnt, ns)).copy()
        if v.shape == (npnt, ns):
            return v
        raise ValueError(
            f"{name} shape {v.shape} incompatible with "
            f"(n_crossings={npnt}, ns_size={ns})"
        )

    a = _coerce(values_outside, "values_outside")
    b = _coerce(values_inside, "values_inside")

    if a is None and b is None:
        z = np.zeros((npnt, ns))
        return z, z.copy()
    if a is None:
        return b.copy(), b
    if b is None:
        return a, a.copy()
    return a, b


def apply_ibm(mat, ibm, values_outside=None, values_inside=None,
              return_bc="vector"):
    """Apply the immersed-boundary method to an operator matrix.

    Ghost columns on both sides of the immersed interface are folded into
    their respective cut-cell rows in a single call.  The matrix is expanded
    from spatial flat indices to full-field flat indices using the ``axes`` /
    ``shape`` information stored in ``ibm``.

    For each wall crossing ``k`` and non-spatial layer ``j``, the matrix
    entry ``v = A[full_row(k,j), full_ghost(k,j)]`` is read, and the row is
    modified with the same geometric Lagrange coefficients (which are
    independent of ``j``).  Wall values may differ per ``j``.

    Parameters
    ----------
    mat : sparse matrix or array
        Operator matrix of shape ``(n_cells, n_cells)`` where
        ``n_cells = ibm.n_cells``.  Converted to CSR internally.
    ibm : IBM
        Immersed-boundary data from :func:`construct_ibm`.
    values_outside : array_like, optional
        Dirichlet wall values for the *outside* (fluid) cut cells.
        Accepted shapes:

        * ``None``: use the same values as ``values_inside``; if both are
          ``None`` the source is zero.
        * scalar or 1-D array of length ``n_crossings``: broadcast over all
          non-spatial layers.
        * 2-D array of shape ``(n_crossings, ns_size)``: one value per
          crossing and non-spatial layer.
    values_inside : array_like, optional
        Dirichlet wall values for the *inside* (solid) cut cells.  Same
        shapes accepted as ``values_outside``.
    return_bc : {'vector', 'matrix'}, optional
        ``'vector'`` (default): return the source vector for the supplied
        wall values.  ``'matrix'``: return the pair of sparse source matrices
        ``(G_out, G_in)`` of shape ``(n_cells, n_crossings * ns_size)`` such
        that the source equals
        ``G_out @ values_outside.ravel() + G_in @ values_inside.ravel()``.

    Returns
    -------
    mat_mod : scipy.sparse.csr_array
        Modified operator matrix, shape ``(n_cells, n_cells)``.
    bc : ndarray or tuple
        When ``return_bc='vector'``: source vector, shape ``(n_cells,)``.
        When ``return_bc='matrix'``: tuple ``(G_out, G_in)`` of
        ``csr_array`` source matrices, shape
        ``(n_cells, n_crossings * ns_size)``.
    """
    n_full = ibm.n_cells
    ns = ibm.ns_size
    npnt = ibm.n_crossings

    val_out, val_in = _normalize_values(values_outside, values_inside, npnt, ns)

    A = csr_array(mat)

    # --- Full flat index expansion ---
    ns_c = _ns_contributions(ibm.shape, ibm.axes)  # (ns,)

    def expand(spatial_flat):
        """Spatial flat → full flat matrix of shape (len, ns)."""
        sc = _spatial_contributions(spatial_flat, ibm.shape, ibm.axes)
        return sc[:, np.newaxis] + ns_c[np.newaxis, :]

    f_row_out = expand(ibm.row_out)       # (npnt, ns)
    f_ghost_out = expand(ibm.ghost_out)   # (npnt, ns)
    f_row_in = expand(ibm.row_in)
    f_ghost_in = expand(ibm.ghost_in)

    has_opp_out = ibm.opp_out >= 0        # (npnt,)
    f_opp_out = expand(np.where(has_opp_out, ibm.opp_out, ibm.row_out))
    has_opp_in = ibm.opp_in >= 0
    f_opp_in = expand(np.where(has_opp_in, ibm.opp_in, ibm.row_in))

    # --- Read ghost matrix entries: v[k, j] = A[row(k,j), ghost(k,j)] ---
    v_out = _read_entries(A, f_row_out.ravel(), f_ghost_out.ravel()).reshape(npnt, ns)
    v_in = _read_entries(A, f_row_in.ravel(), f_ghost_in.ravel()).reshape(npnt, ns)

    def bc_coef(arr):
        """Broadcast 1-D per-crossing array to (npnt, ns) view."""
        return np.broadcast_to(arr[:, np.newaxis], (npnt, ns))

    # --- Matrix correction (COO) ---
    has_opp_out_2d = has_opp_out[:, np.newaxis].repeat(ns, axis=1)  # (npnt, ns) bool
    has_opp_in_2d = has_opp_in[:, np.newaxis].repeat(ns, axis=1)

    cr_out_flat = f_row_out.ravel()
    fg_out_flat = f_ghost_out.ravel()
    fo_out_mask = has_opp_out_2d.ravel()
    cr_in_flat = f_row_in.ravel()
    fg_in_flat = f_ghost_in.ravel()
    fo_in_mask = has_opp_in_2d.ravel()

    cr_out = np.concatenate([cr_out_flat,
                              cr_out_flat[fo_out_mask],
                              cr_out_flat])
    cc_out = np.concatenate([cr_out_flat,                          # diagonal
                              f_opp_out.ravel()[fo_out_mask],      # opposite
                              fg_out_flat])                         # ghost removal
    cd_out = np.concatenate([(v_out * bc_coef(ibm.coef_c_out)).ravel(),
                              (v_out * bc_coef(ibm.coef_o_out)).ravel()[fo_out_mask],
                              -v_out.ravel()])

    cr_in = np.concatenate([cr_in_flat,
                             cr_in_flat[fo_in_mask],
                             cr_in_flat])
    cc_in = np.concatenate([cr_in_flat,
                             f_opp_in.ravel()[fo_in_mask],
                             fg_in_flat])
    cd_in = np.concatenate([(v_in * bc_coef(ibm.coef_c_in)).ravel(),
                              (v_in * bc_coef(ibm.coef_o_in)).ravel()[fo_in_mask],
                              -v_in.ravel()])

    correction = coo_array(
        (np.concatenate([cd_out, cd_in]),
         (np.concatenate([cr_out, cr_in]),
          np.concatenate([cc_out, cc_in]))),
        shape=(n_full, n_full),
    )
    mat_mod = (A + correction.tocsr()).tocsr()

    # --- Row scaling ---
    combined_scale_s = ibm.row_scale_out * ibm.row_scale_in  # (n_spatial_cells,)
    full_scale = _expand_row_scale(combined_scale_s, ibm)
    scaled = np.any(full_scale != 1.0)
    if scaled:
        S = _row_scaling_matrix(full_scale, n_full)
        mat_mod = (S @ mat_mod).tocsr()

    # --- Source matrices G_out and G_in, shape (n_full, npnt * ns) ---
    n_cols = npnt * ns
    # Column index for crossing k and ns layer j: k*ns + j
    k_rep = np.repeat(np.arange(npnt, dtype=np.intp), ns)   # (npnt*ns,)
    j_tile = np.tile(np.arange(ns, dtype=np.intp), npnt)    # (npnt*ns,)
    own_col = k_rep * ns + j_tile                            # (npnt*ns,)

    has_sib_out_flat = np.repeat(ibm.sib_out >= 0, ns)      # (npnt*ns,)
    sib_out_col = (np.where(has_sib_out_flat, np.repeat(ibm.sib_out, ns), 0) * ns + j_tile)
    has_sib_in_flat = np.repeat(ibm.sib_in >= 0, ns)
    sib_in_col = (np.where(has_sib_in_flat, np.repeat(ibm.sib_in, ns), 0) * ns + j_tile)

    G_out = coo_array(
        (np.concatenate([(v_out * bc_coef(ibm.coef_w_out)).ravel(),
                          (v_out * bc_coef(ibm.coef_w_sib_out)).ravel()[has_sib_out_flat]]),
         (np.concatenate([f_row_out.ravel(),
                           f_row_out.ravel()[has_sib_out_flat]]),
          np.concatenate([own_col,
                           sib_out_col[has_sib_out_flat]]))),
        shape=(n_full, n_cols),
    ).tocsr()

    G_in = coo_array(
        (np.concatenate([(v_in * bc_coef(ibm.coef_w_in)).ravel(),
                          (v_in * bc_coef(ibm.coef_w_sib_in)).ravel()[has_sib_in_flat]]),
         (np.concatenate([f_row_in.ravel(),
                           f_row_in.ravel()[has_sib_in_flat]]),
          np.concatenate([own_col,
                           sib_in_col[has_sib_in_flat]]))),
        shape=(n_full, n_cols),
    ).tocsr()

    if scaled:
        G_out = (S @ G_out).tocsr()
        G_in = (S @ G_in).tocsr()

    if return_bc == "matrix":
        return mat_mod, G_out, G_in
    if return_bc == "vector":
        g = (np.asarray(G_out @ val_out.ravel()).ravel()
             + np.asarray(G_in @ val_in.ravel()).ravel())
        return mat_mod, g
    raise ValueError(f"return_bc must be 'vector' or 'matrix', got {return_bc!r}")


def apply_ibm_vector(vec, ibm):
    """Apply the IBM per-row conditioning scale to a flat vector.

    When the operator matrix is constant it is modified once via
    :func:`apply_ibm`; any independent right-hand-side term must be scaled
    by the same per-row factor to keep the system consistent.  This helper
    applies the combined scale for both sides of the interface and expands
    from the spatial grid to the full field.

    Parameters
    ----------
    vec : array_like
        Vector to scale, must have ``size == ibm.n_cells``.  May be shaped
        as the full field or passed as a flat array.
    ibm : IBM
        Immersed-boundary data from :func:`construct_ibm`.

    Returns
    -------
    numpy.ndarray, shape (n_cells,)
        Row-scaled vector, always returned as a 1-D flat array.
    """
    vec = np.asarray(vec, dtype=float).ravel()
    if vec.size != ibm.n_cells:
        raise ValueError(
            f"vec size {vec.size} != n_cells {ibm.n_cells}"
        )
    combined_scale_s = ibm.row_scale_out * ibm.row_scale_in
    full_scale = _expand_row_scale(combined_scale_s, ibm)
    return full_scale * vec
