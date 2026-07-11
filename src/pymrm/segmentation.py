"""SDF-based domain segmentation for immersed boundaries in :mod:`pymrm`.

Immersed boundary conditions are specified per *wall crossing* rather than on a
grid-aligned face, so the spatial broadcasting that domain-wall boundary
conditions enjoy (a single value copied along a wall) is not directly
available.  In practice the most useful spatial pattern is *piecewise constant
per body*: give every dispersed element (a disjoint region of the signed
distance field) its own interface condition.  This module restores that
workflow.

:func:`segment_domain` labels the disjoint regions of the SDF with
:func:`scipy.ndimage.label`.  :func:`crossing_segments` maps each IBM crossing
to the label of the body it bounds, and :func:`segment_values` /
:func:`combine_interface_conditions` expand per-segment data to the
per-crossing arrays consumed by :func:`pymrm.apply_ibm` and
:func:`pymrm.apply_ibm_interface`.  Where a body touches the domain boundary,
:func:`wall_patch` / :func:`wall_values` produce broadcast-ready coefficients
for the ordinary ``{a, b, d}`` wall boundary conditions, so a wall-touching
body can carry the same condition on its wall patch as on its immersed part.
:func:`wall_contact` reports which bodies reach a domain wall — useful on the
fluid side (``region="positive"``) to detect isolated no-flux pockets that
would otherwise make the operator singular.

The ``region`` argument selects which side of the interface is segmented:
``"negative"`` (the default, ``sdf < 0`` solid bodies) or ``"positive"``
(``sdf >= 0`` fluid regions), matching the solid/fluid convention of
:func:`pymrm.construct_ibm`.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label as _ndimage_label
from scipy.ndimage import generate_binary_structure

__all__ = [
    "Segmentation",
    "segment_domain",
    "crossing_segments",
    "segment_values",
    "wall_contact",
    "wall_patch",
    "wall_values",
    "combine_interface_conditions",
    "segment_field",
]


# ---------------------------------------------------------------------------
# Container and construction
# ---------------------------------------------------------------------------

@dataclass
class Segmentation:
    """Per-cell integer labelling of one region of the spatial grid.

    Produced by :func:`segment_domain` (connected components of a signed
    distance field) or by :func:`pymrm.construct_ibm_particles` (one label per
    particle, which keeps touching particles distinct).  The per-segment
    helpers (:func:`crossing_segments`, :func:`segment_values`,
    :func:`combine_interface_conditions`, :func:`wall_patch`,
    :func:`wall_values`, :func:`segment_field`) work with either source.

    Attributes
    ----------
    labels : ndarray of int
        Integer label field with the shape of the spatial grid.  ``0`` marks
        the *other* region; the labelled region carries ``1 .. n_segments``.
    n_segments : int
        Number of disjoint segments.
    region : {'negative', 'positive'}
        Which side of the interface was labelled (``sdf < 0`` or ``sdf >= 0``).
    connectivity : int
        Connectivity used for labelling (``1`` = faces, ``labels.ndim`` =
        including diagonals).  Reported as ``1`` for the particle path.
    sizes : ndarray of int, shape (n_segments,)
        Number of cells in each segment; ``sizes[s - 1]`` is the size of the
        segment with label ``s``.
    """

    labels: np.ndarray
    n_segments: int
    region: str
    connectivity: int
    sizes: np.ndarray


def segment_domain(sdf, *, region="negative", connectivity=1):
    """Label the disjoint regions of a signed distance field.

    Parameters
    ----------
    sdf : array_like
        Signed distance field sampled at the spatial cell centres, as passed to
        :func:`pymrm.construct_ibm`.
    region : {'negative', 'positive'}, optional
        Segment ``sdf < 0`` (solid bodies, default) or ``sdf >= 0`` (fluid
        regions).
    connectivity : int, optional
        Neighbour connectivity for labelling, ``1 <= connectivity <= sdf.ndim``.
        ``1`` (default) links face neighbours only, matching the staircase
        boundary of the IBM crossings; ``sdf.ndim`` also links diagonal
        neighbours.

    Returns
    -------
    Segmentation
    """
    sdf = np.asarray(sdf)
    if sdf.ndim == 0:
        raise ValueError("sdf must be an array with at least one spatial axis")
    if region == "negative":
        mask = sdf < 0.0
    elif region == "positive":
        mask = ~(sdf < 0.0)
    else:
        raise ValueError(
            f"region must be 'negative' or 'positive', got {region!r}")
    if not 1 <= connectivity <= sdf.ndim:
        raise ValueError(
            f"connectivity must be in 1..{sdf.ndim}, got {connectivity}")

    structure = generate_binary_structure(sdf.ndim, connectivity)
    labels, n = _ndimage_label(mask, structure=structure)
    sizes = np.bincount(labels.ravel(),
                        minlength=n + 1)[1:].astype(np.intp)
    return Segmentation(labels=labels, n_segments=int(n), region=region,
                        connectivity=int(connectivity), sizes=sizes)


# ---------------------------------------------------------------------------
# Crossing association
# ---------------------------------------------------------------------------

def _check_spatial_match(seg, ibm):
    if tuple(seg.labels.shape) != tuple(ibm.spatial_shape):
        raise ValueError(
            f"segmentation labels shape {tuple(seg.labels.shape)} does not "
            f"match ibm.spatial_shape {tuple(ibm.spatial_shape)}")


def crossing_segments(seg, ibm):
    """Segment label of the body bounded by each IBM crossing.

    For ``region="negative"`` the label of the *inside* (solid) cut cell is
    returned; for ``region="positive"`` the *outside* (fluid) cut cell.  In
    either case the cut cell lies in the segmented region, so every returned
    label is in ``1 .. n_segments``.

    Parameters
    ----------
    seg : Segmentation
    ibm : IBM

    Returns
    -------
    ndarray of int, shape (n_crossings,)
    """
    _check_spatial_match(seg, ibm)
    if ibm.n_crossings == 0:
        return np.empty(0, dtype=np.intp)
    rows = ibm.row_in if seg.region == "negative" else ibm.row_out
    out = seg.labels.ravel()[rows].astype(np.intp)
    if np.any(out == 0):
        raise RuntimeError(
            "a crossing mapped to background label 0; this should not happen "
            "for a segmentation of the same side as the crossings — please "
            "report")
    return out


# ---------------------------------------------------------------------------
# Per-segment -> per-crossing / per-cell expansion
# ---------------------------------------------------------------------------

def _segment_lookup(values, n_segments, default, name, trailing_shape=None):
    """Build a ``(n_segments + 1, *trailing)`` lookup table indexed by label.

    Row ``s`` holds the value for segment ``s``; row ``0`` (background) holds
    *default*.  ``provided`` flags which rows carry a value.  ``values`` is
    either an array of shape ``(n_segments, *trailing)`` or a ``{label: value}``
    dict; per-segment values are broadcast to a common trailing shape (forced to
    *trailing_shape* when given).
    """
    if isinstance(values, dict):
        items = {}
        for label, v in values.items():
            label = int(label)
            if not 1 <= label <= n_segments:
                raise ValueError(
                    f"{name}: segment label {label} out of range "
                    f"1..{n_segments}")
            items[label] = np.asarray(v, dtype=float)
        shapes = [a.shape for a in items.values()]
        if default is not None:
            shapes.append(np.asarray(default, dtype=float).shape)
        if trailing_shape is not None:
            shapes.append(tuple(trailing_shape))
        trailing = np.broadcast_shapes(*shapes) if shapes else ()
        lookup = np.zeros((n_segments + 1,) + trailing, dtype=float)
        provided = np.zeros(n_segments + 1, dtype=bool)
        if default is not None:
            lookup[:] = np.broadcast_to(np.asarray(default, dtype=float),
                                        trailing)
            provided[:] = True
        for label, a in items.items():
            lookup[label] = np.broadcast_to(a, trailing)
            provided[label] = True
        return lookup, provided

    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0 or arr.shape[0] != n_segments:
        raise ValueError(
            f"{name}: array must have shape (n_segments={n_segments}, ...), "
            f"got {arr.shape}")
    trailing = arr.shape[1:]
    if trailing_shape is not None:
        trailing = np.broadcast_shapes(trailing, tuple(trailing_shape))
        arr = np.broadcast_to(arr, (n_segments,) + trailing)
    lookup = np.zeros((n_segments + 1,) + trailing, dtype=float)
    provided = np.ones(n_segments + 1, dtype=bool)
    if default is not None:
        lookup[0] = np.broadcast_to(np.asarray(default, dtype=float), trailing)
    else:
        provided[0] = False
    lookup[1:] = arr
    return lookup, provided


def _index_lookup(lookup, provided, idx, name):
    """Index *lookup* by integer label array *idx*, checking coverage."""
    idx = np.asarray(idx, dtype=np.intp)
    flat = idx.ravel()
    if not provided[flat].all():
        missing = sorted(set(flat[~provided[flat]].tolist()))
        raise ValueError(
            f"{name}: no value for segment label(s) {missing} and no default "
            f"given")
    return lookup[idx]


def _to_point_shape(arr, ns_ndim):
    """Right-pad ns singleton axes so *arr* broadcasts as canonical point data.

    ``arr`` has shape ``(npnt, *trailing)``; the result has shape
    ``(npnt, 1, ..., 1, *trailing)`` with a total of ``ns_ndim`` trailing axes,
    keeping the crossing axis leading so it aligns with ``(npnt, *ns_shape)``.
    """
    arr = np.asarray(arr, dtype=float)
    trailing = arr.shape[1:]
    if len(trailing) > ns_ndim:
        raise ValueError(
            f"per-segment value has {len(trailing)} trailing axes, more than "
            f"the {ns_ndim} non-spatial axes of the field")
    pad = (1,) * (ns_ndim - len(trailing))
    return arr.reshape((arr.shape[0],) + pad + trailing)


def segment_values(values, seg, ibm, *, default=None):
    """Expand per-segment values to a per-crossing array for :func:`pymrm.apply_ibm`.

    Parameters
    ----------
    values : array_like or dict
        Either an array of shape ``(n_segments, *trailing)`` or a
        ``{label: value}`` dict.  The trailing dimensions must be broadcastable
        to ``ibm.ns_shape`` (the non-spatial axes).
    seg : Segmentation
    ibm : IBM
    default : optional
        Value for segments absent from a ``dict`` input.  If ``None`` a missing
        label that carries crossings raises ``ValueError``.

    Returns
    -------
    ndarray, shape ``(n_crossings, 1, ..., 1, *trailing)``
        Canonical point-value array, ready to pass as ``values_outside`` /
        ``values_inside``.
    """
    seg_ids = crossing_segments(seg, ibm)
    lookup, provided = _segment_lookup(values, seg.n_segments, default,
                                       "segment_values")
    picked = _index_lookup(lookup, provided, seg_ids, "segment_values")
    return _to_point_shape(picked, len(ibm.ns_shape))


def segment_field(values, seg, *, default=0.0):
    """Expand per-segment values to a per-cell spatial field.

    ``lookup[seg.labels]`` — background cells (label 0) receive *default*.  For
    example a per-particle diffusivity for the cell-centred-``D`` conjugate
    diffusion pattern.

    Parameters
    ----------
    values : array_like or dict
        ``(n_segments, *trailing)`` array or ``{label: value}`` dict.
    seg : Segmentation
    default : optional
        Value for background (label 0) cells; ``0.0`` by default.

    Returns
    -------
    ndarray, shape ``seg.labels.shape + trailing``
    """
    lookup, provided = _segment_lookup(values, seg.n_segments, default,
                                       "segment_field")
    return _index_lookup(lookup, provided, seg.labels, "segment_field")


# ---------------------------------------------------------------------------
# Domain-wall contact and patches
# ---------------------------------------------------------------------------

def wall_contact(seg):
    """Whether each segment reaches each domain wall.

    Returns
    -------
    ndarray of bool, shape (n_segments, ndim_spatial, 2)
        ``out[s - 1, a, 0]`` / ``out[s - 1, a, 1]`` is ``True`` when segment
        ``s`` has cells in the first / last layer along spatial axis ``a``.

    Notes
    -----
    A fluid-side segmentation (``region="positive"``) whose segment touches no
    wall is an isolated pocket; with all-Neumann surroundings it makes the
    operator singular.  Detect them with
    ``np.flatnonzero(~wall_contact(seg).any(axis=(1, 2))) + 1``.
    """
    labels = seg.labels
    out = np.zeros((seg.n_segments, labels.ndim, 2), dtype=bool)
    for a in range(labels.ndim):
        for si, layer in enumerate((0, -1)):
            face = np.take(labels, layer, axis=a)
            present = np.unique(face)
            present = present[present > 0]
            out[present - 1, a, si] = True
    return out


def _spatial_axis(ibm, axis, side):
    if axis not in tuple(ibm.axes):
        raise ValueError(
            f"axis {axis} is not a spatial axis; spatial axes are "
            f"{tuple(ibm.axes)}")
    if side not in ("lower", "upper"):
        raise ValueError(f"side must be 'lower' or 'upper', got {side!r}")
    return tuple(ibm.axes).index(axis), (0 if side == "lower" else -1)


def wall_patch(seg, ibm, axis, side):
    """Segment labels on one domain wall, shaped as a full-field coefficient.

    Parameters
    ----------
    seg : Segmentation
    ibm : IBM
    axis : int
        Full-field axis of the wall (must be one of ``ibm.axes``), matching the
        ``axis`` argument of the boundary-condition operators.
    side : {'lower', 'upper'}
        Which end of that axis, matching the ``(bc_lower, bc_upper)`` tuple.

    Returns
    -------
    ndarray of int
        Label of the boundary cell (``0`` = other region) reshaped to full field
        rank with the wall ``axis`` and every non-spatial axis at size 1.  Use
        directly in ``np.where`` to build a ``{a, b, d}`` wall coefficient, e.g.
        ``np.where(wall_patch(seg, ibm, 0, "lower") == k, value_k, other)``.
    """
    _check_spatial_match(seg, ibm)
    sax, layer = _spatial_axis(ibm, axis, side)
    face = np.take(seg.labels, layer, axis=sax)
    full_shape = tuple(
        ibm.shape[a] if (a in tuple(ibm.axes) and a != axis) else 1
        for a in range(len(ibm.shape)))
    return face.reshape(full_shape).astype(np.intp)


def wall_values(values, seg, ibm, axis, side, *, default=0.0):
    """Per-segment values on one domain wall as a full-field BC coefficient.

    Combines :func:`wall_patch` with a per-segment lookup so that a
    wall-touching body can be given the same condition on its wall patch as on
    its immersed boundary.  The per-segment values may carry non-spatial
    structure (broadcastable to ``ibm.ns_shape``).

    Parameters
    ----------
    values : array_like or dict
        ``(n_segments, *trailing)`` array or ``{label: value}`` dict, trailing
        broadcastable to ``ibm.ns_shape``.
    seg, ibm, axis, side : see :func:`wall_patch`.
    default : optional
        Value for cells not in the segmented region (label 0); ``0.0`` default.

    Returns
    -------
    ndarray
        Full field shape with the wall ``axis`` at size 1 — a ready ``a`` / ``b``
        / ``d`` coefficient for :func:`pymrm.construct_grad` and friends.
    """
    _check_spatial_match(seg, ibm)
    sax, layer = _spatial_axis(ibm, axis, side)
    face = np.take(seg.labels, layer, axis=sax)
    lookup, provided = _segment_lookup(values, seg.n_segments, default,
                                       "wall_values",
                                       trailing_shape=ibm.ns_shape)
    picked = _index_lookup(lookup, provided, face, "wall_values")

    # picked axes: [other spatial (ascending full-axis order), *ns_shape].
    # Reinsert the reduced spatial axis, then move to interleaved field order.
    spatial_sorted = sorted(ibm.axes)
    ns_sorted = [a for a in range(len(ibm.shape)) if a not in tuple(ibm.axes)]
    grouped = np.expand_dims(picked, spatial_sorted.index(axis))
    dest = spatial_sorted + ns_sorted
    return np.moveaxis(grouped, range(grouped.ndim), dest)


# ---------------------------------------------------------------------------
# Per-segment interface conditions
# ---------------------------------------------------------------------------

def _ic_slot(ic, eq, key, side):
    """Extract one coefficient (scalar/array) from a two-dict ic tuple."""
    d = ic[eq] if ic[eq] else {}
    if key == "d":
        return d.get("d", 0.0)
    pair = d.get(key, (0.0, 0.0))
    return pair[side]


def combine_interface_conditions(ic_by_segment, seg, ibm, *, default=None):
    """Merge per-segment interface conditions into one per-crossing ``ic``.

    Each entry of ``ic_by_segment`` is a two-dict interface condition in the
    format of :func:`pymrm.apply_ibm_interface`.  The returned ``ic`` has, for
    every coefficient slot, a per-crossing array assembled from the owning
    segment of each crossing — so all crossings of one body share that body's
    condition.

    Parameters
    ----------
    ic_by_segment : dict
        ``{label: ic}`` mapping segment label to a two-dict interface condition.
        Per-segment coefficients must be scalar or broadcastable to
        ``ibm.ns_shape`` (no nested per-crossing arrays).
    seg : Segmentation
    ibm : IBM
    default : ic tuple, optional
        Interface condition for segments absent from ``ic_by_segment``.  If
        ``None`` a missing label that carries crossings raises ``ValueError``.

    Returns
    -------
    tuple
        A single ``ic`` usable with :func:`pymrm.apply_ibm_interface` /
        :func:`pymrm.construct_ibm_interface_values`.
    """
    seg_ids = crossing_segments(seg, ibm)
    labels = sorted(int(k) for k in ic_by_segment)
    ns_ndim = len(ibm.ns_shape)

    def slot(eq, key, side):
        vals = {L: np.asarray(_ic_slot(ic_by_segment[L], eq, key, side),
                              dtype=float) for L in labels}
        dflt = (None if default is None
                else np.asarray(_ic_slot(default, eq, key, side), dtype=float))
        name = f"ic[{eq}]['{key}']" + ("" if side is None else f"[{side}]")
        lookup, provided = _segment_lookup(vals, seg.n_segments, dflt, name)
        picked = _index_lookup(lookup, provided, seg_ids, name)
        return _to_point_shape(picked, ns_ndim)

    ic0 = {"a": (slot(0, "a", 0), slot(0, "a", 1)),
           "b": (slot(0, "b", 0), slot(0, "b", 1)),
           "d": slot(0, "d", None)}
    ic1 = {"a": (slot(1, "a", 0), slot(1, "a", 1)),
           "b": (slot(1, "b", 0), slot(1, "b", 1)),
           "d": slot(1, "d", None)}
    return (ic0, ic1)
