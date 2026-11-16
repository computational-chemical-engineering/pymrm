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

The classification and wall positions may equally be supplied by an assembly
of analytic shapes rather than a sampled field: :func:`pymrm.construct_ibm_particles`
produces the very same :class:`IBM` object (with exact per-particle wall
positions and normals) and everything downstream — :func:`apply_ibm`,
:mod:`pymrm.ibm_recon`, :mod:`pymrm.ibm_coupling` — is unchanged.

Multi-dimensional fields
------------------------
The field on which the IBM operator acts may have *non-spatial* axes
(components, phases, species, etc.) in addition to the spatial axes specified
by the ``axes`` argument to :func:`construct_ibm`.

Per-crossing data (Dirichlet wall values, interface-condition coefficients)
follows a *canonical point shape* ``(n_crossings, *ns_shape)`` — the field
shape with the spatial axes removed and the crossing axis leading.  Any input
NumPy can broadcast to that shape is accepted, mirroring how wall boundary
conditions broadcast over the non-spatial axes; see
:func:`_normalize_point_values`.

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
contributions are **independent** (an outer sum over crossings and ns layers).

Sign convention
---------------
The modified matrix ``M`` and source ``g`` satisfy ``value = M @ c + g``
(source is *added*), matching the gradient/divergence operator convention.
The optional per-row conditioning scale is folded identically into the matrix
and the source.
"""

from dataclasses import dataclass
import math
from typing import NamedTuple
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


def _is_pure_spatial(ibm):
    """True when full-field flat indices equal spatial flat indices.

    This holds when there are no non-spatial axes and the spatial axes are the
    leading axes in canonical order, so the spatial→full index expansion is the
    identity and can be skipped entirely.
    """
    return ibm.ns_size == 1 and ibm.axes == tuple(range(len(ibm.spatial_shape)))


def _expand_full(ibm, spatial_flat):
    """Expand spatial flat indices to full-field flat indices.

    Returns an array of shape ``(len(spatial_flat), ns_size)``.  Only the cells
    actually needed are expanded (never the whole grid), and the common purely
    spatial case is a no-op reshape.
    """
    spatial_flat = np.asarray(spatial_flat, dtype=np.intp)
    if _is_pure_spatial(ibm):
        return spatial_flat[:, np.newaxis]
    sc = _spatial_contributions(spatial_flat, ibm.shape, ibm.axes)
    ns_c = _ns_contributions(ibm.shape, ibm.axes)
    return sc[:, np.newaxis] + ns_c[np.newaxis, :]


def _combined_cut_scale(ibm):
    """Full-field rows and factors of the per-row IBM conditioning scale.

    Only cut-cell rows differ from unity, so this returns just those rows
    (already expanded to full-field flat indices) and their scale factors,
    rather than a dense length-``n_cells`` vector.

    Returns
    -------
    rows : ndarray of intp, shape (n_scaled,)
        Full-field flat row indices whose scale differs from 1.
    factors : ndarray of float, shape (n_scaled,)
        Matching scale factors.
    """
    combined = ibm.row_scale_out * ibm.row_scale_in      # (n_spatial_cells,)
    cut = np.flatnonzero(combined != 1.0)
    if cut.size == 0:
        return np.empty(0, dtype=np.intp), np.empty(0)
    rows = _expand_full(ibm, cut)                        # (n_cut, ns)
    factors = np.broadcast_to(combined[cut][:, np.newaxis], rows.shape)
    return rows.ravel(), factors.ravel()


# ---------------------------------------------------------------------------
# Internal geometry construction
# ---------------------------------------------------------------------------

def _neighbor(arr, axis, offset):
    """Return the axis-shifted copy of *arr* and a boolean validity mask.

    The shifted-in (invalid) border is left uninitialised: every caller masks
    those cells out with the returned ``valid`` array, so their values are
    never read.  Avoiding a full ``arr.copy()`` and ``np.ones`` allocation
    roughly halves the memory traffic of this routine, which dominates
    :func:`construct_ibm` on large 3-D grids.
    """
    n = arr.shape[axis]
    out = np.empty_like(arr)
    valid = np.zeros(arr.shape, dtype=bool)
    dst = [slice(None)] * arr.ndim
    src = [slice(None)] * arr.ndim
    if offset > 0:
        dst[axis] = slice(0, n - offset)
        src[axis] = slice(offset, n)
    else:
        o = -offset
        dst[axis] = slice(o, n)
        src[axis] = slice(0, n - o)
    dst = tuple(dst)
    out[dst] = arr[tuple(src)]
    valid[dst] = True
    return out, valid


def _lagrange3(n0, n1, n2, p):
    """Quadratic Lagrange basis values at *p* over nodes (*n0*, *n1*, *n2*)."""
    l0 = (p - n1) * (p - n2) / ((n0 - n1) * (n0 - n2))
    l1 = (p - n0) * (p - n2) / ((n1 - n0) * (n1 - n2))
    l2 = (p - n0) * (p - n1) / ((n2 - n0) * (n2 - n1))
    return l0, l1, l2


def _sdf_theta_fn(sdf, strides):
    """Directed face-crossing fractions from a signed-distance field.

    Returns a callable ``theta_fn(cells, axis, direction)`` giving, for each
    cut cell, the fractional wall position ``theta in (0, 1]`` along the
    segment from the cell centre towards its neighbour at
    ``cells + direction * strides[axis]``, from linear interpolation of the
    SDF (clipped to ``[_THETA_MIN, _THETA_MAX]``).
    """
    flat_sdf = np.asarray(sdf, dtype=float).ravel()

    def theta_fn(cells, axis, direction):
        sdf_c = flat_sdf[cells]
        sdf_g = flat_sdf[cells + direction * strides[axis]]
        return np.clip(sdf_c / (sdf_c - sdf_g), _THETA_MIN, _THETA_MAX)

    return theta_fn


# Per-crossing columns collected for one side, and the dtype of each when empty.
_SIDE_COLUMNS = (
    "row", "ghost", "opp", "coef_c", "coef_o",
    "coef_w_self", "coef_w_sib", "is_sw",
    "axis", "direction", "crossing_key", "coords",
)
_SIDE_DTYPES = ((np.intp,) * 3 + (float,) * 4 + (bool,) + (np.intp,) * 3
                + (float,))


class _SideGrid(NamedTuple):
    """Spatial grid of an IBM side: shape, cell-centre coordinates, flat strides."""

    spatial_shape: tuple
    x_c: list
    strides: np.ndarray


class _SideRegions(NamedTuple):
    """Cells that own rows on a side (*region*) and the opposite region (*other*)."""

    region: np.ndarray
    other: np.ndarray


class _CutMasks(NamedTuple):
    """Masks of one axis direction: ghost cells, sandwich cells, opposite validity."""

    gmask: np.ndarray
    omask: np.ndarray
    opp_valid: np.ndarray


class _CutGeometry(NamedTuple):
    """Cut cells of one axis direction with the coordinates along that axis."""

    cells: np.ndarray
    strides: np.ndarray
    a: int
    direction: int
    xs: tuple           # (xc, xg, xo, xw)


class _CutKinds(NamedTuple):
    """Reconstruction kind of every cut cell."""

    sandwich: np.ndarray
    fallback: np.ndarray
    normal: np.ndarray


def _cut_coefficients(theta_fn, geo, kinds):
    """Lagrange coefficients and opposite cells of the cut cells of one direction."""
    cells, a, direction = geo.cells, geo.a, geo.direction
    xc, xg, xo, xw = geo.xs
    sandwich, fallback, normal = kinds
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
        opp_lin[normal] = cells[normal] - direction * geo.strides[a]

    if np.any(sandwich):
        # Opposite neighbour is in-domain for sandwich cells (omask ⊆
        # valid on the opposite side); its wall sits on the opposite
        # face, i.e. the crossing in the -direction.
        theta_o = theta_fn(cells[sandwich], a, -direction)
        xw_o = xc[sandwich] + theta_o * (xo[sandwich] - xc[sandwich])
        m0, m1, m2 = _lagrange3(xw_o, xc[sandwich], xw[sandwich], xg[sandwich])
        coef_w_sib[sandwich] = m0
        coef_c[sandwich] = m1
        coef_w_self[sandwich] = m2

    if np.any(fallback):
        xcf, xwf, xgf = xc[fallback], xw[fallback], xg[fallback]
        coef_c[fallback] = (xgf - xwf) / (xcf - xwf)
        coef_w_self[fallback] = (xgf - xcf) / (xwf - xcf)
    return coef_c, coef_o, coef_w_self, coef_w_sib, opp_lin


def _cut_cell_entries(theta_fn, grid, a, direction, masks):
    """Crossing entries of one axis direction, or ``None`` without cut cells.

    Returns ``(entries, has_fallback)``: a dict of per-crossing columns and
    whether any cut cell needs the first-order fallback reconstruction.
    """
    spatial_shape, x_c, strides = grid
    ndim_s = len(spatial_shape)
    cells = np.flatnonzero(masks.gmask.ravel())
    if cells.size == 0:
        return None
    multi = np.unravel_index(cells, spatial_shape)
    ia = multi[a]

    # Ghost neighbour is in-domain for every cut cell (gmask ⊆ valid).
    theta = theta_fn(cells, a, direction)

    xc = x_c[a][ia]
    xg = x_c[a][ia + direction]
    # The opposite cell at ``ia - direction`` feeds only the 'normal'
    # (3-point) and 'sandwich' reconstructions, both of which require it
    # to be in-domain (``opp_valid`` / ``omask``).  For 'fallback' cut
    # cells the solid sits against the domain boundary, so ``ia -
    # direction`` is out of bounds and ``xo`` is never used; clip the
    # gather index to keep it in range instead of indexing past the edge.
    xo = x_c[a][np.clip(ia - direction, 0, spatial_shape[a] - 1)]
    xw = xc + theta * (xg - xc)

    sandwich = masks.omask.ravel()[cells]
    fallback = (~sandwich) & (~masks.opp_valid.ravel()[cells])
    normal = (~sandwich) & (~fallback)

    geo = _CutGeometry(cells, strides, a, direction, (xc, xg, xo, xw))
    coef_c, coef_o, coef_w_self, coef_w_sib, opp_lin = _cut_coefficients(
        theta_fn, geo, _CutKinds(sandwich, fallback, normal))

    ghost_lin = cells + direction * strides[a]
    lower = np.minimum(cells, ghost_lin)
    key = lower * ndim_s + a

    coords = np.empty((cells.size, ndim_s))
    for j in range(ndim_s):
        coords[:, j] = x_c[j][multi[j]]
    coords[:, a] = xw

    entries = {
        "row": cells.astype(np.intp),
        "ghost": ghost_lin.astype(np.intp),
        "opp": opp_lin,
        "coef_c": coef_c,
        "coef_o": coef_o,
        "coef_w_self": coef_w_self,
        "coef_w_sib": coef_w_sib,
        "is_sw": sandwich,
        "axis": np.full(cells.size, a, dtype=np.intp),
        "direction": np.full(cells.size, direction, dtype=np.intp),
        "crossing_key": key.astype(np.intp),
        "coords": coords,
    }
    return entries, bool(np.any(fallback))


def _axis_entries(theta_fn, grid, regions, a):
    """Crossing entries along axis *a* for both directions (empty ones dropped)."""
    # Only the region masks are shifted (cheap bool arrays); the wall
    # positions are queried per cut cell through ``theta_fn`` further down,
    # avoiding two full float-array shifts per axis.
    other_p, valid_p = _neighbor(regions.other, a, +1)
    other_m, valid_m = _neighbor(regions.other, a, -1)
    region = regions.region
    found = []
    for direction in (+1, -1):
        if direction == +1:
            fwd, bwd = (valid_p, other_p), (valid_m, other_m)
        else:
            fwd, bwd = (valid_m, other_m), (valid_p, other_p)
        masks = _CutMasks(region & fwd[0] & fwd[1], region & bwd[0] & bwd[1],
                          bwd[0])
        entries = _cut_cell_entries(theta_fn, grid, a, direction, masks)
        if entries is not None:
            found.append(entries)
    return found


def _concat_columns(cols, ndim_s):
    """Concatenate the collected per-direction columns (empty arrays if none)."""
    if cols["row"]:
        return {k: np.concatenate(v, axis=0) for k, v in cols.items()}
    merged = {k: np.empty(0, dtype=d)
              for k, d in zip(_SIDE_COLUMNS, _SIDE_DTYPES)}
    merged[_SIDE_COLUMNS[-1]] = np.empty((0, ndim_s))   # the coordinates
    return merged


def _link_siblings(row, axis, direction_arr, is_sw, ndim_s):
    """Link sandwich siblings within one side.

    A sibling is the entry with the same (row, axis) but the opposite
    direction; find it by matching a packed integer key with a sorted search
    instead of a Python-level dict loop.
    """
    sib = np.full(row.size, -1, dtype=np.intp)
    if np.any(is_sw):
        dir_bit = (direction_arr > 0).astype(np.intp)
        base = (row * ndim_s + axis) * 2
        key_self = base + dir_bit             # this entry's key
        key_want = base + (1 - dir_bit)       # its opposite-direction sibling
        order = np.argsort(key_self, kind="stable")
        keys_sorted = key_self[order]
        sw = np.flatnonzero(is_sw)
        pos = np.searchsorted(keys_sorted, key_want[sw])
        in_range = pos < keys_sorted.size
        match = np.zeros(sw.size, dtype=bool)
        match[in_range] = keys_sorted[pos[in_range]] == key_want[sw][in_range]
        sib[sw[match]] = order[pos[match]]
    return sib


def _row_scale(n_spatial, row, coefs, rescale):
    """Per-row conditioning scale (per spatial cell)."""
    row_scale = np.ones(n_spatial)
    if rescale and row.size:
        max_coef = np.maximum.reduce([np.abs(c) for c in coefs])
        per_cell = np.zeros(n_spatial)
        np.maximum.at(per_cell, row, max_coef)
        cut = per_cell > 1.0
        row_scale[cut] = 1.0 / per_cell[cut]
    return row_scale


def _build_side(theta_fn, grid, regions, rescale):
    """Build an :class:`_IBMSide` for one region of the immersed interface.

    Parameters
    ----------
    theta_fn : callable
        ``theta_fn(cells, axis, direction) -> theta`` giving the fractional
        wall position along the segment from each cut cell towards its
        region-switching neighbour (see :func:`_sdf_theta_fn`).  Queried only
        for faces between *region* and *other* cells.
    grid : _SideGrid
        Spatial grid shape, cell-centre coordinates (one 1-D array per
        spatial axis) and C-order flat strides.
    regions : _SideRegions
        ``region``: cells that own rows on this side; ``other``: cells in the
        opposite region (ghost cells).
    rescale : bool
        Whether to compute per-row conditioning scale.
    """
    ndim_s = len(grid.spatial_shape)
    n_spatial = math.prod(grid.spatial_shape)

    cols = {k: [] for k in _SIDE_COLUMNS}

    warned = False
    for a in range(ndim_s):
        for entries, has_fallback in _axis_entries(theta_fn, grid, regions, a):
            if has_fallback and not warned:
                warnings.warn(
                    "IBM: immersed solid adjacent to the domain boundary; "
                    "falling back to first-order reconstruction.",
                    RuntimeWarning, stacklevel=4,
                )
                warned = True
            for k, v in entries.items():
                cols[k].append(v)

    m = _concat_columns(cols, ndim_s)
    row, ghost, opp = m["row"], m["ghost"], m["opp"]
    coef_c, coef_o = m["coef_c"], m["coef_o"]
    coef_w_self, coef_w_sib = m["coef_w_self"], m["coef_w_sib"]
    is_sw, axis = m["is_sw"], m["axis"]
    direction_arr, crossing_key = m["direction"], m["crossing_key"]
    coords = m["coords"]
    sib = _link_siblings(row, axis, direction_arr, is_sw, ndim_s)
    row_scale = _row_scale(
        n_spatial, row, (coef_c, coef_o, coef_w_self, coef_w_sib), rescale)

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


def _scale_csr_rows(mat, rows, factors):
    """Multiply selected CSR rows of *mat* by per-row *factors*, in place.

    This replaces a full diagonal ``S @ mat`` product (which touches every
    stored entry) by writing only the non-zeros of the handful of scaled rows,
    turning an ``O(nnz)`` operation into ``O(nnz in scaled rows)``.  *mat* must
    be a CSR array that owns its ``data`` buffer.
    """
    if rows.size == 0:
        return
    indptr = mat.indptr
    lengths = indptr[rows + 1] - indptr[rows]
    total = int(lengths.sum())
    if total == 0:
        return
    # Vectorised gather of the non-zero positions of the selected rows.
    seg_end = np.cumsum(lengths)
    offsets = np.arange(total) - np.repeat(seg_end - lengths, lengths)
    nnz_idx = np.repeat(indptr[rows], lengths) + offsets
    mat.data[nnz_idx] *= np.repeat(factors, lengths)


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
    strides = np.array(
        [math.prod(sdf.shape[a + 1:]) for a in range(sdf.ndim)], dtype=np.intp
    )
    out_s, in_s, meta = _construct_ibm_sides(
        sdf < 0.0, _sdf_theta_fn(sdf, strides), x_c,
        _FieldLayout(axes, shape), rescale)
    return _pair_sides_to_ibm(out_s, in_s, meta)


class _FieldLayout(NamedTuple):
    """Spatial axes of the full field and the full field shape (either may be None)."""

    axes: tuple
    shape: tuple


class _IBMMeta(NamedTuple):
    """Grid and field-layout metadata shared by both sides of an IBM."""

    spatial_shape: tuple
    shape: tuple
    axes: tuple
    ns_shape: tuple
    ns_size: int
    n_cells: int
    n_spatial_cells: int


def _normalize_core_x_c(x_c, ndim_s):
    """Normalise x_c to a list of 1-D arrays, one per spatial axis."""
    if isinstance(x_c, np.ndarray) and x_c.ndim == 1:
        x_c = [x_c]
    else:
        x_c = [np.asarray(xci, dtype=float) for xci in x_c]
    if len(x_c) != ndim_s:
        raise ValueError(
            f"len(x_c)={len(x_c)} must equal the spatial dimension {ndim_s}"
        )
    return x_c


def _normalize_axes_shape(layout, spatial_shape):
    """Normalise the spatial axes and the full field shape."""
    axes, shape = layout
    if axes is None and shape is None:
        axes = tuple(range(len(spatial_shape)))
        shape = spatial_shape
    elif shape is None:
        axes = tuple(int(a) for a in axes)
        shape = spatial_shape
    else:
        axes = tuple(int(a) for a in axes)
        shape = tuple(shape)
        if tuple(shape[a] for a in axes) != spatial_shape:
            raise ValueError(
                f"spatial shape {spatial_shape} is inconsistent with "
                f"shape={shape} at axes={axes}"
            )
    return axes, shape


def _ibm_meta(spatial_shape, shape, axes):
    """Non-spatial shape, sizes and layout of the full field."""
    axes_set = set(axes)
    ns_axes = [a for a in range(len(shape)) if a not in axes_set]
    ns_shape = tuple(shape[a] for a in ns_axes)
    ns_size = math.prod(ns_shape) if ns_shape else 1
    n_cells = math.prod(shape)
    n_spatial_cells = math.prod(spatial_shape)
    return _IBMMeta(spatial_shape=tuple(spatial_shape), shape=shape, axes=axes,
                    ns_shape=ns_shape, ns_size=ns_size, n_cells=n_cells,
                    n_spatial_cells=n_spatial_cells)


def _construct_ibm_sides(solid, theta_fn, x_c, layout, rescale):
    """Unpaired IBM construction: ``(out_side, in_side, meta)``.

    ``solid`` is the boolean cell-classification on the spatial grid;
    ``theta_fn(cells, axis, direction)`` supplies the fractional wall position
    for directed region-switching faces (see :func:`_build_side`).  A caller
    can append extra crossings (e.g. particle contact crossings) before
    :func:`_pair_sides_to_ibm`.
    """
    solid = np.asarray(solid, dtype=bool)
    ndim_s = solid.ndim
    spatial_shape = solid.shape
    x_c = _normalize_core_x_c(x_c, ndim_s)
    axes, shape = _normalize_axes_shape(layout, spatial_shape)

    strides = np.array(
        [math.prod(spatial_shape[a + 1:]) for a in range(ndim_s)], dtype=np.intp
    )

    fluid = ~solid

    grid = _SideGrid(spatial_shape, x_c, strides)
    out_s = _build_side(theta_fn, grid, _SideRegions(fluid, solid), rescale)
    in_s = _build_side(theta_fn, grid, _SideRegions(solid, fluid), rescale)

    return out_s, in_s, _ibm_meta(spatial_shape, shape, axes)


def _pair_sides_to_ibm(out_s, in_s, meta):
    """Pair the two :class:`_IBMSide` objects into an :class:`IBM`.

    Outside and inside entries are matched by sorting on the canonical face
    key; every face must appear exactly once on each side.
    """
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

        spatial_shape=tuple(meta.spatial_shape),
        shape=meta.shape,
        axes=meta.axes,
        ns_shape=meta.ns_shape,
        ns_size=meta.ns_size,
        n_cells=meta.n_cells,
        n_spatial_cells=meta.n_spatial_cells,
    )


# ---------------------------------------------------------------------------
# Application helpers
# ---------------------------------------------------------------------------

def _read_entries(mat, rows, cols):
    """Return ``mat[rows[k], cols[k]]`` for every ``k`` as a 1-D array."""
    if rows.size == 0:
        return np.empty(0, dtype=float)
    return np.asarray(mat[rows, cols]).ravel()


def _normalize_point_values(value, npnt, ns_shape, name):
    """Normalise a per-crossing quantity to a float array of shape ``(npnt, ns_size)``.

    Point data behaves exactly like an array of *canonical* shape
    ``(npnt, *ns_shape)`` — the field shape with the spatial axes removed and
    the crossing axis leading.  Any input NumPy can broadcast to that shape is
    accepted, and nothing else (strict NumPy semantics).  Examples for a field
    with ``ns_shape == (np, nc)``:

    * scalar → same value everywhere;
    * ``(nc,)`` → per component;
    * ``(np, 1)`` → per phase;
    * ``(np, nc)`` → per phase-and-component;
    * ``(npnt, 1, 1)`` → per crossing;
    * ``(npnt, np, nc)`` → fully specified.

    The result is materialised and C-order reshaped to ``(npnt, ns_size)``,
    where ``ns_size`` flattens ``ns_shape`` in C order — matching the layer
    index ``j`` used throughout this module.

    Note that a bare 1-D array of length ``npnt`` is **not** treated as
    per-crossing when non-spatial axes are present (it would collide with a
    ``(nc,)`` per-component array); reshape to ``(npnt, 1, ..., 1)`` instead.
    """
    ns_size = math.prod(ns_shape) if ns_shape else 1
    target = (npnt,) + tuple(ns_shape)
    v = np.asarray(value, dtype=float)
    try:
        b = np.broadcast_to(v, target)
    except ValueError:
        hint = ""
        if v.ndim == 1 and v.size == npnt and ns_shape:
            idx = ", ".join(["None"] * len(ns_shape))
            singleton = (npnt,) + (1,) * len(ns_shape)
            hint = (
                f"; a 1-D array of length n_crossings={npnt} is not treated as "
                f"per-crossing when non-spatial axes are present — reshape to "
                f"{singleton} (e.g. value[:, {idx}]) for per-crossing values"
            )
        raise ValueError(
            f"{name}: shape {v.shape} is not broadcastable to the canonical "
            f"point shape (n_crossings, *ns_shape) = {target}{hint}"
        ) from None
    return np.ascontiguousarray(b).reshape(npnt, ns_size)


def _normalize_values(values_outside, values_inside, npnt, ns_shape):
    """Normalise IBM wall values for both sides to shape ``(npnt, ns_size)``.

    Each non-``None`` value is broadcast to the canonical point shape
    ``(npnt, *ns_shape)`` by :func:`_normalize_point_values`.  ``None``
    handling: both ``None`` → zeros; one ``None`` → mirror the other.
    """
    ns_size = math.prod(ns_shape) if ns_shape else 1
    a = (None if values_outside is None
         else _normalize_point_values(values_outside, npnt, ns_shape,
                                      "values_outside"))
    b = (None if values_inside is None
         else _normalize_point_values(values_inside, npnt, ns_shape,
                                      "values_inside"))

    if a is None and b is None:
        z = np.zeros((npnt, ns_size))
        return z, z.copy()
    if a is None:
        return b.copy(), b
    if b is None:
        return a, a.copy()
    return a, b


class _SideIndices(NamedTuple):
    """Full-field flat indices of one side's cut-cell rows, ghosts and opposites."""

    row: np.ndarray
    ghost: np.ndarray
    opp: np.ndarray
    has_opp: np.ndarray


class _SourceCols(NamedTuple):
    """Source-matrix column indices of a crossing layer and the matrix shape."""

    own_col: np.ndarray
    j_tile: np.ndarray
    shape: tuple


def _make_expander(ibm):
    """Return ``expand(spatial_flat)`` mapping spatial to full flat indices."""
    pure_spatial = _is_pure_spatial(ibm)
    ns_c = _ns_contributions(ibm.shape, ibm.axes)  # (ns,)

    def expand(spatial_flat):
        """Spatial flat → full flat matrix of shape (len, ns)."""
        if pure_spatial:
            return np.asarray(spatial_flat, dtype=np.intp)[:, np.newaxis]
        sc = _spatial_contributions(spatial_flat, ibm.shape, ibm.axes)
        return sc[:, np.newaxis] + ns_c[np.newaxis, :]

    return expand


def _side_indices(expand, row, ghost, opp):
    """Expanded row, ghost and opposite indices of one side."""
    f_row = expand(row)                   # (npnt, ns)
    f_ghost = expand(ghost)               # (npnt, ns)
    has_opp = opp >= 0                    # (npnt,)
    f_opp = expand(np.where(has_opp, opp, row))
    return _SideIndices(f_row, f_ghost, f_opp, has_opp)


def _per_crossing(arr, shape):
    """Broadcast 1-D per-crossing array to (npnt, ns) view."""
    return np.broadcast_to(arr[:, np.newaxis], shape)


def _side_correction(v, idx, coef_c, coef_o):
    """COO triplets of the matrix correction that folds one side's ghosts."""
    ns = v.shape[1]
    has_opp_2d = idx.has_opp[:, np.newaxis].repeat(ns, axis=1)  # (npnt, ns) bool
    cr_flat = idx.row.ravel()
    fg_flat = idx.ghost.ravel()
    fo_mask = has_opp_2d.ravel()
    cr = np.concatenate([cr_flat,
                         cr_flat[fo_mask],
                         cr_flat])
    cc = np.concatenate([cr_flat,                          # diagonal
                         idx.opp.ravel()[fo_mask],        # opposite
                         fg_flat])                         # ghost removal
    cd = np.concatenate([(v * _per_crossing(coef_c, v.shape)).ravel(),
                         (v * _per_crossing(coef_o, v.shape)).ravel()[fo_mask],
                         -v.ravel()])
    return cr, cc, cd


def _side_source_matrix(v, idx, cols, sib, coefs):
    """Source matrix of one side; *coefs* are the wall and sibling-wall coefficients."""
    ns = v.shape[1]
    has_sib_flat = np.repeat(sib >= 0, ns)
    sib_col = (np.where(has_sib_flat, np.repeat(sib, ns), 0) * ns + cols.j_tile)
    return coo_array(
        (np.concatenate([(v * _per_crossing(coefs[0], v.shape)).ravel(),
                         (v * _per_crossing(coefs[1], v.shape)).ravel()[has_sib_flat]]),
         (np.concatenate([idx.row.ravel(),
                          idx.row.ravel()[has_sib_flat]]),
          np.concatenate([cols.own_col,
                          sib_col[has_sib_flat]]))),
        shape=cols.shape,
    ).tocsr()


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
        Dirichlet wall values for the *outside* (fluid) cut cells.  ``None``
        (the default) uses the same values as ``values_inside``; if both are
        ``None`` the source is zero.  Otherwise any array broadcastable to the
        canonical point shape ``(n_crossings, *ns_shape)`` is accepted (strict
        NumPy semantics), e.g. a scalar, a ``(nc,)`` per-component array, a
        ``(n_crossings, 1, ..., 1)`` per-crossing array, or the fully specified
        ``(n_crossings, *ns_shape)``.  See :func:`_normalize_point_values`.
        A bare 1-D array of length ``n_crossings`` is only per-crossing for a
        purely spatial field; when non-spatial axes are present reshape it to
        ``(n_crossings, 1, ..., 1)``.
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
    The number of return values depends on *return_bc*:

    * ``return_bc='vector'`` → ``(mat_mod, g)``:

      - ``mat_mod`` : ``scipy.sparse.csr_array``, shape ``(n_cells, n_cells)``
        -- the modified operator matrix;
      - ``g`` : ndarray, shape ``(n_cells,)`` -- the source vector for the
        supplied wall values (``value = mat_mod @ c + g``).

    * ``return_bc='matrix'`` → ``(mat_mod, G_out, G_in)``:

      - ``mat_mod`` : as above;
      - ``G_out``, ``G_in`` : ``csr_array``, shape
        ``(n_cells, n_crossings * ns_size)`` -- source matrices with
        ``g = G_out @ values_outside.ravel() + G_in @ values_inside.ravel()``.
    """
    n_full = ibm.n_cells
    ns = ibm.ns_size
    npnt = ibm.n_crossings

    val_out, val_in = _normalize_values(values_outside, values_inside, npnt,
                                        ibm.ns_shape)

    A = csr_array(mat)

    # --- Full flat index expansion ---
    expand = _make_expander(ibm)
    out = _side_indices(expand, ibm.row_out, ibm.ghost_out, ibm.opp_out)
    inn = _side_indices(expand, ibm.row_in, ibm.ghost_in, ibm.opp_in)

    # --- Read ghost matrix entries: v[k, j] = A[row(k,j), ghost(k,j)] ---
    v_out = _read_entries(A, out.row.ravel(), out.ghost.ravel()).reshape(npnt, ns)
    v_in = _read_entries(A, inn.row.ravel(), inn.ghost.ravel()).reshape(npnt, ns)

    # --- Matrix correction (COO) ---
    cr_out, cc_out, cd_out = _side_correction(v_out, out, ibm.coef_c_out,
                                              ibm.coef_o_out)
    cr_in, cc_in, cd_in = _side_correction(v_in, inn, ibm.coef_c_in,
                                           ibm.coef_o_in)

    correction = coo_array(
        (np.concatenate([cd_out, cd_in]),
         (np.concatenate([cr_out, cr_in]),
          np.concatenate([cc_out, cc_in]))),
        shape=(n_full, n_full),
    )
    mat_mod = (A + correction.tocsr()).tocsr()

    # --- Row scaling (only cut-cell rows differ from unity) ---
    scale_rows, scale_vals = _combined_cut_scale(ibm)
    _scale_csr_rows(mat_mod, scale_rows, scale_vals)

    # --- Source matrices G_out and G_in, shape (n_full, npnt * ns) ---
    n_cols = npnt * ns
    # Column index for crossing k and ns layer j: k*ns + j
    k_rep = np.repeat(np.arange(npnt, dtype=np.intp), ns)   # (npnt*ns,)
    j_tile = np.tile(np.arange(ns, dtype=np.intp), npnt)    # (npnt*ns,)
    own_col = k_rep * ns + j_tile                            # (npnt*ns,)
    cols = _SourceCols(own_col, j_tile, (n_full, n_cols))

    G_out = _side_source_matrix(v_out, out, cols, ibm.sib_out,
                                (ibm.coef_w_out, ibm.coef_w_sib_out))
    G_in = _side_source_matrix(v_in, inn, cols, ibm.sib_in,
                               (ibm.coef_w_in, ibm.coef_w_sib_in))

    _scale_csr_rows(G_out, scale_rows, scale_vals)
    _scale_csr_rows(G_in, scale_rows, scale_vals)

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
    out = vec.copy()
    scale_rows, scale_vals = _combined_cut_scale(ibm)
    out[scale_rows] *= scale_vals
    return out


def reconstruct_ghost_values(ibm, x_c, field, wall_values=0.0, side="out",
                             theta_min=0.25):
    """Reconstruct per-crossing ghost-cell values of a cell-centered field.

    For every wall crossing the value of ``field`` at the *ghost* cell center
    (the first cell on the other side of the immersed interface) is
    extrapolated along the crossing axis from the same-side cells and the
    Dirichlet wall value.  This is useful for evaluating state-dependent
    coefficients (e.g. composition-dependent diffusivities) at faces near the
    immersed boundary, where neighbouring cell values are missing.

    The ghost value of a field is direction dependent (one solid cell can be
    the ghost of several crossings, with slightly different reconstructions);
    values are therefore returned per crossing.  Use
    :func:`fill_ghost_values` for a filled (averaged) copy of the field.

    Reconstruction: second-order Lagrange through the cut cell, its opposite
    same-side neighbour, and the wall point, evaluated at the ghost center.
    When the wall lies within ``theta_min`` of the cut-cell center (measured
    as a fraction of the center-to-ghost distance), the cut-cell value is
    skipped and a first-order (linear) reconstruction through the opposite
    neighbour and the wall point is used instead: the second-order formula is
    an unstable extrapolation when two of its nodes nearly coincide.  Without
    an opposite neighbour the reconstruction is linear through the cut cell
    and the wall, or the wall value itself when additionally
    ``theta < theta_min``.

    Parameters
    ----------
    ibm : IBM
        Immersed-boundary data from :func:`construct_ibm`.
    x_c : array_like or list of array_like
        Cell-center coordinates per spatial axis (same as ``construct_ibm``).
    field : array_like
        Cell-centered field.  Shape must start with ``ibm.spatial_shape``;
        trailing (non-spatial) dimensions are allowed and handled
        elementwise.
    wall_values : array_like, optional
        Dirichlet wall values; broadcastable to
        ``(n_crossings, *trailing_shape)``.
    side : {'out', 'in'}, optional
        Reconstruct from the fluid side (``'out'``, default) or solid side.
    theta_min : float, optional
        Threshold on the wall position fraction below which the
        reconstruction switches from second to first order.

    Returns
    -------
    ghost_index : ndarray of int, shape (n_crossings,)
        Spatial flat index of each ghost cell.
    ghost_values : ndarray, shape (n_crossings, *trailing_shape)
        Reconstructed field values at the ghost-cell centers.
    """
    nd = len(ibm.spatial_shape)
    if isinstance(x_c, np.ndarray) and x_c.ndim == 1 and nd == 1:
        x_c = [np.asarray(x_c, dtype=float)]
    else:
        x_c = [np.asarray(xc, dtype=float) for xc in x_c]

    rows = getattr(ibm, f"row_{side}")
    ghosts = getattr(ibm, f"ghost_{side}")
    opps = getattr(ibm, f"opp_{side}")
    npnt = ibm.n_crossings

    field = np.asarray(field, dtype=float)
    trailing = field.shape[nd:]
    f_flat = field.reshape(ibm.n_spatial_cells, -1)
    w = np.broadcast_to(
        np.asarray(wall_values, dtype=float),
        (npnt, *trailing)).reshape(npnt, -1)

    def _coord_along_axis(flat_idx):
        idx = np.unravel_index(np.maximum(flat_idx, 0), ibm.spatial_shape)
        per_axis = np.stack([x_c[a][idx[a]] for a in range(nd)], axis=0)
        return per_axis[ibm.axis, np.arange(npnt)]

    x_row = _coord_along_axis(rows)
    x_ghost = _coord_along_axis(ghosts)
    x_opp = _coord_along_axis(opps)
    x_w = ibm.coords[np.arange(npnt), ibm.axis]

    theta = (x_w - x_row) / (x_ghost - x_row)
    has_opp = opps >= 0
    use_first = theta < theta_min

    L_row = np.zeros(npnt)
    L_opp = np.zeros(npnt)
    L_w = np.zeros(npnt)

    # second order: Lagrange through (x_opp, x_row, x_w) at x_ghost
    m = has_opp & ~use_first
    if m.any():
        xo, xr, xw, xg = x_opp[m], x_row[m], x_w[m], x_ghost[m]
        L_opp[m] = (xg - xr) * (xg - xw) / ((xo - xr) * (xo - xw))
        L_row[m] = (xg - xo) * (xg - xw) / ((xr - xo) * (xr - xw))
        L_w[m] = (xg - xo) * (xg - xr) / ((xw - xo) * (xw - xr))
    # first order through (x_opp, x_w): skip the too-close cut cell
    m = has_opp & use_first
    if m.any():
        xo, xw, xg = x_opp[m], x_w[m], x_ghost[m]
        L_opp[m] = (xg - xw) / (xo - xw)
        L_w[m] = (xg - xo) / (xw - xo)
    # no opposite neighbour: linear through (x_row, x_w) ...
    m = ~has_opp & ~use_first
    if m.any():
        xr, xw, xg = x_row[m], x_w[m], x_ghost[m]
        L_row[m] = (xg - xw) / (xr - xw)
        L_w[m] = (xg - xr) / (xw - xr)
    # ... or the wall value itself when the cut cell is also too close
    m = ~has_opp & use_first
    L_w[m] = 1.0

    f_row = f_flat[rows]
    f_opp = f_flat[np.maximum(opps, 0)]
    vals = (L_row[:, None] * f_row + L_opp[:, None] * f_opp
            + L_w[:, None] * w)
    return ghosts.copy(), vals.reshape(npnt, *trailing)


def fill_ghost_values(ibm, x_c, field, wall_values=0.0, side="out",
                      theta_min=0.25):
    """Copy of ``field`` with interface-adjacent ghost cells filled.

    Ghost cells shared by several crossings receive the average of the
    per-crossing reconstructions of :func:`reconstruct_ghost_values`; all
    other cells keep their original values.  The result is suitable for
    interpolating cell-centered state variables to faces near the immersed
    boundary (e.g. with :func:`pymrm.interp_cntr_to_stagg`).
    """
    ghosts, vals = reconstruct_ghost_values(
        ibm, x_c, field, wall_values=wall_values, side=side,
        theta_min=theta_min)
    field = np.asarray(field, dtype=float)
    out = field.reshape(ibm.n_spatial_cells, -1).copy()
    vals2 = vals.reshape(vals.shape[0], -1)
    sums = np.zeros_like(out)
    counts = np.zeros(out.shape[0])
    np.add.at(sums, ghosts, vals2)
    np.add.at(counts, ghosts, 1.0)
    filled = counts > 0
    out[filled] = sums[filled] / counts[filled, None]
    return out.reshape(field.shape)
