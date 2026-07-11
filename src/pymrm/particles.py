"""Particle-based immersed boundaries for :mod:`pymrm`.

Instead of sampling one global signed-distance field, an assembly of
:class:`Particle` objects generates the immersed-boundary data directly:
each particle classifies the cells it covers (bounding-box window only),
provides the *exact* wall position on every cut face (analytic where the
shape allows it, root-finding otherwise), and evaluates the *exact* outward
normal at every wall crossing.  This avoids the two error sources of the
union-SDF route near particle contacts:

* the union ``min_i(level_i)`` has a gradient kink on the contact medial
  axis, degrading SDF-gradient normals to O(1) locally (observed to reduce
  the flux/conjugate IBM from 2nd to ~1st order for touching particles);
* the fractional wall position ``theta`` interpolated from the union is
  polluted when the ghost cell's union value comes from a *different*
  particle.

Particle protocol
-----------------
A particle answers three geometric questions, all in world coordinates:

* :meth:`Particle.level` — signed level function (< 0 inside, > 0 outside;
  approximately a distance near the surface),
* :meth:`Particle.intersect` — surface crossing on a straight segment whose
  endpoints straddle the surface,
* :meth:`Particle.normal` — outward (solid → fluid) unit normal at surface
  points.

The base class supplies world↔body transforms (``position`` plus an
``orientation``: an angle in 2-D, a :class:`scipy.spatial.transform.Rotation`
in 3-D) and numeric defaults for ``intersect`` (vectorised bisection) and
``normal`` (finite differences), so a new shape only has to implement the
body-frame level function and bounding box.  :class:`Sphere` (alias
:class:`Circle`), :class:`Box`, :class:`AnalyticParticle` and
:class:`GridParticle` (B-spline interpolated local samples) are provided.

Contact policy
--------------
When two particles are closer than one grid cell, the cell-centre
classification alone cannot see the gap: two adjacent cells are solid but
belong to *different* particles.  ``construct_ibm_particles`` detects these
*contact faces* (the segment between the two cell centres leaves one particle
before entering the other) and applies a policy:

* ``"no_flux"`` (default): a contact crossing is created with each side's
  Lagrange reconstruction anchored at its *own* particle surface; impose a
  Neumann–Neumann interface condition on these crossings (see
  :func:`contact_conditions`) so there is no direct transport between the
  particles.
* ``"merge"``: no crossing is created; the particles are numerically
  connected (the union-SDF behaviour).
* ``"error"``: raise, for setups where contact indicates a bug.

Faces where the particles genuinely overlap (the segment does not leave one
particle before entering the other) are always merged — interpenetrating
particles form one body.

Usage
-----
::

    particles = [Sphere(c, r) for c, r in zip(centers, radii)]
    ibm, info = construct_ibm_particles(particles, x_c)
    recon = construct_ibm_normal_derivative(ibm, info.pseudo_sdf, x_c,
                                         normals=info.normals)
    ic = contact_conditions(base_ic, ibm, info)
    A, g = apply_ibm_interface(L, ibm, recon, ic)
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from pymrm.ibm import (
    _THETA_MIN,
    _THETA_MAX,
    _IBMSide,
    _construct_ibm_core,
    _lagrange3,
    _neighbor,
    _pair_sides_to_ibm,
)
from pymrm.segmentation import Segmentation

__all__ = [
    "Particle",
    "Sphere",
    "Circle",
    "Box",
    "AnalyticParticle",
    "GridParticle",
    "ParticleIBMInfo",
    "construct_ibm_particles",
    "contact_conditions",
]


# ---------------------------------------------------------------------------
# Orientation helpers
# ---------------------------------------------------------------------------

def _rotation_matrix(orientation, ndim):
    """Body→world rotation matrix from an orientation specification.

    ``None`` → identity (returned as ``None`` to skip the transform).
    2-D: a scalar angle in radians.  3-D: a
    :class:`scipy.spatial.transform.Rotation` (quaternion-backed).  Any
    dimension: an explicit ``(ndim, ndim)`` rotation matrix.
    """
    if orientation is None:
        return None
    if np.ndim(orientation) == 2:
        R = np.asarray(orientation, dtype=float)
        if R.shape != (ndim, ndim):
            raise ValueError(
                f"orientation matrix shape {R.shape} != ({ndim}, {ndim})")
        return R
    if np.ndim(orientation) == 0 and not hasattr(orientation, "as_matrix"):
        if ndim != 2:
            raise ValueError(
                "a scalar orientation (angle) is only valid in 2-D; use a "
                "scipy.spatial.transform.Rotation in 3-D")
        c, s = math.cos(float(orientation)), math.sin(float(orientation))
        return np.array([[c, -s], [s, c]])
    if hasattr(orientation, "as_matrix"):
        R = np.asarray(orientation.as_matrix(), dtype=float)
        if R.shape != (ndim, ndim):
            raise ValueError(
                f"Rotation is {R.shape[0]}-D but the particle is {ndim}-D")
        return R
    raise TypeError(f"unsupported orientation: {orientation!r}")


# ---------------------------------------------------------------------------
# Particle protocol
# ---------------------------------------------------------------------------

class Particle(ABC):
    """Abstract particle: a shape at a position with an orientation.

    Subclasses implement the **body-frame** interface
    (:meth:`level_body`, :meth:`bounding_box_body`, optionally
    :meth:`normal_body`); the world-frame API used by the IBM assembly
    (:meth:`level`, :meth:`normal`, :meth:`intersect`,
    :meth:`bounding_box`) is provided here, including the numeric fallbacks.

    Parameters
    ----------
    position : array_like, shape (ndim,)
        World position of the particle (body-frame origin).
    orientation : optional
        ``None`` (default), an angle in radians (2-D), a
        :class:`scipy.spatial.transform.Rotation` (3-D), or an explicit
        rotation matrix.
    """

    def __init__(self, position, orientation=None):
        self.position = np.asarray(position, dtype=float)
        if self.position.ndim != 1:
            raise ValueError("position must be a 1-D coordinate array")
        self.ndim = self.position.size
        self._R = _rotation_matrix(orientation, self.ndim)
        self.orientation = orientation

    # -- body-frame interface (implemented by shapes) -----------------------

    @abstractmethod
    def level_body(self, coords):
        """Signed level function at body-frame ``coords`` shaped (..., ndim)."""

    @abstractmethod
    def bounding_box_body(self):
        """Body-frame bounding box ``((lo, hi), ...)`` per axis."""

    def normal_body(self, coords):
        """Gradient direction of :meth:`level_body` (finite differences)."""
        coords = np.asarray(coords, dtype=float)
        eps = 1e-5 * max(hi - lo for lo, hi in self.bounding_box_body())
        g = np.empty(coords.shape)
        for a in range(self.ndim):
            dp = coords.copy(); dp[..., a] += eps
            dm = coords.copy(); dm[..., a] -= eps
            g[..., a] = (self.level_body(dp) - self.level_body(dm)) / (2 * eps)
        return g

    # -- world-frame transforms ---------------------------------------------

    def to_body(self, coords):
        v = np.asarray(coords, dtype=float) - self.position
        return v if self._R is None else v @ self._R

    def vec_to_world(self, vecs):
        return vecs if self._R is None else vecs @ self._R.T

    # -- world-frame API consumed by the assembly ---------------------------

    def level(self, coords):
        """Signed level function at world ``coords`` shaped (..., ndim)."""
        return self.level_body(self.to_body(coords))

    def normal(self, coords):
        """Outward (solid→fluid) unit normal at world surface points."""
        n = self.vec_to_world(self.normal_body(self.to_body(coords)))
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def bounding_box(self, pad=0.0):
        """World axis-aligned bounding box ``((lo, hi), ...)`` per axis."""
        box = self.bounding_box_body()
        corners = np.array(np.meshgrid(*box, indexing="ij")).reshape(self.ndim, -1).T
        world = (corners if self._R is None else corners @ self._R.T) + self.position
        return tuple((world[:, a].min() - pad, world[:, a].max() + pad)
                     for a in range(self.ndim))

    def intersect(self, p0, p1):
        """Surface crossing fraction ``t`` on the segments ``p0 → p1``.

        ``p0``/``p1`` are ``(n, ndim)`` batches whose endpoints straddle the
        surface (``level(p0)`` and ``level(p1)`` of opposite sign); returns
        ``t in (0, 1)`` with ``level(p0 + t (p1 - p0)) == 0``.  Default:
        vectorised bisection on :meth:`level`; shapes with closed-form
        intersections override this.
        """
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        a = np.zeros(p0.shape[0])
        b = np.ones(p0.shape[0])
        fa = self.level(p0)
        d = p1 - p0
        for _ in range(60):
            m = 0.5 * (a + b)
            fm = self.level(p0 + m[:, None] * d)
            same = (fm < 0) == (fa < 0)
            a = np.where(same, m, a)
            fa = np.where(same, fm, fa)
            b = np.where(same, b, m)
        return 0.5 * (a + b)


class Sphere(Particle):
    """Sphere (any dimension; in 2-D this is a disk — see :class:`Circle`).

    Fully analytic: exact level function, normals, and segment intersections.
    """

    def __init__(self, center, radius):
        super().__init__(center)
        self.radius = float(radius)

    def level_body(self, coords):
        return np.linalg.norm(np.asarray(coords, dtype=float), axis=-1) - self.radius

    def normal_body(self, coords):
        return np.asarray(coords, dtype=float)

    def bounding_box_body(self):
        r = self.radius
        return tuple((-r, r) for _ in range(self.ndim))

    def intersect(self, p0, p1):
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        d = p1 - p0
        m = p0 - self.position
        a = np.sum(d * d, axis=-1)
        b = np.sum(d * m, axis=-1)
        c = np.sum(m * m, axis=-1) - self.radius**2
        disc = np.sqrt(np.maximum(b * b - a * c, 0.0))
        t1 = (-b - disc) / a
        t2 = (-b + disc) / a
        # Endpoints straddle the surface: entering picks the first root,
        # exiting (p0 inside) the second.
        return np.where(c > 0.0, t1, t2)


Circle = Sphere


class Box(Particle):
    """Axis-aligned (in body frame) box; rotate via ``orientation``.

    ``half_extents`` are the half side lengths per axis.  Exact SDF and
    face normals; segment intersections by the default bisection.
    """

    def __init__(self, position, half_extents, orientation=None):
        position = np.asarray(position, dtype=float)
        super().__init__(position, orientation)
        self.half_extents = np.broadcast_to(
            np.asarray(half_extents, dtype=float), (self.ndim,)).copy()

    def level_body(self, coords):
        q = np.abs(np.asarray(coords, dtype=float)) - self.half_extents
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.max(q, axis=-1), 0.0)
        return outside + inside

    def normal_body(self, coords):
        coords = np.asarray(coords, dtype=float)
        q = np.abs(coords) - self.half_extents
        pos = np.maximum(q, 0.0)
        out = np.any(q > 0.0, axis=-1)
        n = np.where(out[..., None], pos,
                     np.where(q == np.max(q, axis=-1, keepdims=True), 1.0, 0.0))
        return n * np.sign(coords + np.where(coords == 0.0, 1.0, 0.0))

    def bounding_box_body(self):
        return tuple((-h, h) for h in self.half_extents)


class AnalyticParticle(Particle):
    """Particle from a user-supplied body-frame level function.

    Parameters
    ----------
    level_func : callable
        ``level_func(coords) -> values`` with ``coords`` shaped (..., ndim);
        negative inside, positive outside, approximately a distance near the
        surface.
    bounding_box : tuple of (lo, hi)
        Body-frame box containing the particle surface.
    position, orientation : see :class:`Particle`.
    normal_func : callable, optional
        Body-frame outward normal direction (need not be normalised);
        default: finite differences of *level_func*.
    """

    def __init__(self, level_func, bounding_box, position, orientation=None,
                 normal_func=None):
        super().__init__(position, orientation)
        self._level_func = level_func
        self._box = tuple((float(lo), float(hi)) for lo, hi in bounding_box)
        if len(self._box) != self.ndim:
            raise ValueError("bounding_box length must equal len(position)")
        self._normal_func = normal_func

    def level_body(self, coords):
        return np.asarray(self._level_func(np.asarray(coords, dtype=float)))

    def normal_body(self, coords):
        if self._normal_func is None:
            return super().normal_body(coords)
        return np.asarray(self._normal_func(np.asarray(coords, dtype=float)))

    def bounding_box_body(self):
        return self._box


class GridParticle(Particle):
    """Particle from level-function samples on its own body-frame grid.

    The samples are interpolated with a cubic B-spline
    (:class:`scipy.interpolate.RegularGridInterpolator`), so the particle can
    be translated and rotated for free.  The local grid **must extend beyond
    the particle surface** (positive samples all around); queries outside the
    local grid return the clamped boundary value plus the clamping distance,
    keeping the sign correct far away.

    Parameters
    ----------
    values : ndarray
        Level-function samples, negative inside.
    x_local : list of 1-D arrays
        Body-frame cell coordinates of the sample grid, one per axis.
    position, orientation : see :class:`Particle`.
    method : str, optional
        Interpolation method (default ``"cubic"``).
    """

    def __init__(self, values, x_local, position, orientation=None,
                 method="cubic"):
        position = np.asarray(position, dtype=float)
        super().__init__(position, orientation)
        from scipy.interpolate import RegularGridInterpolator
        values = np.asarray(values, dtype=float)
        x_local = [np.asarray(x, dtype=float) for x in x_local]
        if values.ndim != self.ndim or len(x_local) != self.ndim:
            raise ValueError("values/x_local dimensionality mismatch")
        self._lo = np.array([x[0] for x in x_local])
        self._hi = np.array([x[-1] for x in x_local])
        self._interp = RegularGridInterpolator(
            x_local, values, method=method, bounds_error=False, fill_value=None)
        self._eps = 1e-5 * float(np.max(self._hi - self._lo))

    def level_body(self, coords):
        coords = np.asarray(coords, dtype=float)
        clipped = np.clip(coords, self._lo, self._hi)
        excess = np.linalg.norm(coords - clipped, axis=-1)
        return self._interp(clipped) + excess

    def normal_body(self, coords):
        coords = np.asarray(coords, dtype=float)
        eps = self._eps
        g = np.empty(coords.shape)
        for a in range(self.ndim):
            dp = coords.copy(); dp[..., a] += eps
            dm = coords.copy(); dm[..., a] -= eps
            g[..., a] = (self.level_body(dp) - self.level_body(dm)) / (2 * eps)
        return g

    def bounding_box_body(self):
        return tuple((lo, hi) for lo, hi in zip(self._lo, self._hi))


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

@dataclass
class ParticleIBMInfo:
    """Side-car information produced by :func:`construct_ibm_particles`.

    Attributes
    ----------
    owner : ndarray of int
        Per spatial cell, index of the owning particle (deepest level
        function) or ``-1`` for fluid.  Shaped like the spatial grid.
    crossing_particle : ndarray of int, shape (n_crossings,)
        Owning particle of the solid (``in``) side of each crossing.
    contact : ndarray of bool, shape (n_crossings,)
        True for contact crossings (solid–solid faces between two particles
        under the ``"no_flux"`` policy).  For these,
        ``crossing_particle`` is the ``in``-side particle and
        ``contact_partner`` the ``out``-side one.
    contact_partner : ndarray of int, shape (n_crossings,)
        The ``out``-side particle of a contact crossing, ``-1`` elsewhere.
    normals : ndarray, shape (n_crossings, ndim)
        Exact outward (solid→fluid) unit normals from the owning particle —
        pass to ``construct_ibm_normal_derivative(..., normals=...)``.  For
        contact crossings: the ``in``-side particle's outward normal.
    pseudo_sdf : ndarray
        Sign-correct union level field on the spatial grid (positive filler
        far from every particle) for region classification in the
        reconstruction.
    segmentation : Segmentation
        Per-particle labels (``owner + 1``) — valid even at exact contact,
        where :func:`pymrm.segment_domain` would merge the bodies.
    """

    owner: np.ndarray
    crossing_particle: np.ndarray
    contact: np.ndarray
    contact_partner: np.ndarray
    normals: np.ndarray
    pseudo_sdf: np.ndarray
    segmentation: Segmentation


def _normalize_x_c(x_c):
    if isinstance(x_c, np.ndarray) and x_c.ndim == 1:
        return [np.asarray(x_c, dtype=float)]
    return [np.asarray(x, dtype=float) for x in x_c]


def _window(particle, x_c, halo):
    """Index window (slices) of the particle's bounding box + halo cells."""
    slices = []
    for a, (lo, hi) in enumerate(particle.bounding_box()):
        i0 = int(np.searchsorted(x_c[a], lo, side="left")) - halo
        i1 = int(np.searchsorted(x_c[a], hi, side="right")) + halo
        i0 = max(i0, 0)
        i1 = min(i1, x_c[a].size)
        if i1 <= i0:
            return None
        slices.append(slice(i0, i1))
    return tuple(slices)


def _window_flat(slices, spatial_shape, strides):
    """Flat spatial indices of a window, shaped like the window."""
    ndim = len(spatial_shape)
    idx = 0
    for a in range(ndim):
        r = np.arange(slices[a].start, slices[a].stop, dtype=np.intp)
        shape = [1] * ndim
        shape[a] = r.size
        idx = idx + (r * strides[a]).reshape(shape)
    return idx


def _window_coords(slices, x_c):
    """(nw..., ndim) world coordinates of the window's cell centres."""
    grids = np.meshgrid(*[x_c[a][slices[a]] for a in range(len(x_c))],
                        indexing="ij")
    return np.stack(grids, axis=-1)


class _FaceThetaMap:
    """Vectorised lookup for precomputed directed face thetas."""

    def __init__(self, keys, thetas, ndim):
        order = np.argsort(keys, kind="stable")
        self._keys = keys[order]
        self._thetas = thetas[order]
        self._ndim = ndim

    @staticmethod
    def pack(cells, axis, direction, ndim):
        return (cells * ndim + axis) * 2 + (np.asarray(direction) > 0)

    def __call__(self, cells, axis, direction):
        key = self.pack(np.asarray(cells, dtype=np.intp), axis, direction,
                        self._ndim)
        pos = np.searchsorted(self._keys, key)
        ok = (pos < self._keys.size)
        ok &= self._keys[np.minimum(pos, self._keys.size - 1)] == key
        if not np.all(ok):
            raise RuntimeError(
                "particle IBM: wall-position query for an unregistered face; "
                "this is a bug — please report")
        return np.clip(self._thetas[pos], _THETA_MIN, _THETA_MAX)


def _cell_coords(cells, x_c, spatial_shape):
    multi = np.unravel_index(cells, spatial_shape)
    return np.column_stack([x_c[a][multi[a]] for a in range(len(x_c))])


def construct_ibm_particles(particles, x_c, *, axes=None, shape=None,
                            rescale=True, contact="no_flux", halo=2,
                            fill_value=None):
    """Build immersed-boundary data directly from a particle assembly.

    Parameters
    ----------
    particles : sequence of Particle
    x_c : array_like or list of array_like
        Cell-centre coordinates, one 1-D array per spatial axis.
    axes, shape, rescale : see :func:`pymrm.construct_ibm`.
    contact : {'no_flux', 'merge', 'error'}, optional
        Policy for solid–solid faces between two different particles whose
        surfaces are separated by a sub-cell gap (see the module docstring).
        Genuinely overlapping particles are always merged.
    halo : int, optional
        Extra cells around each particle's bounding box when classifying.
    fill_value : float, optional
        ``pseudo_sdf`` value for cells not covered by any particle window
        (default: 4× the largest cell spacing — any positive value works,
        only the sign is used downstream).

    Returns
    -------
    ibm : IBM
        Standard immersed-boundary container; all of :func:`pymrm.apply_ibm`,
        :func:`pymrm.apply_ibm_interface`, etc. apply unchanged.
    info : ParticleIBMInfo
        Ownership, exact normals, contact bookkeeping, ``pseudo_sdf`` and a
        per-particle :class:`~pymrm.segmentation.Segmentation`.
    """
    x_c = _normalize_x_c(x_c)
    spatial_shape = tuple(x.size for x in x_c)
    ndim = len(spatial_shape)
    n_spatial = math.prod(spatial_shape)
    strides = np.array([math.prod(spatial_shape[a + 1:]) for a in range(ndim)],
                       dtype=np.intp)
    if contact not in ("no_flux", "merge", "error"):
        raise ValueError(
            f"contact must be 'no_flux', 'merge' or 'error', got {contact!r}")
    for p in particles:
        if p.ndim != ndim:
            raise ValueError(
                f"particle dimension {p.ndim} != grid dimension {ndim}")
    if fill_value is None:
        fill_value = 4.0 * max(float(np.max(np.diff(x))) if x.size > 1 else 1.0
                               for x in x_c)

    # --- classification: owner (deepest level) + pseudo union level --------
    owner = np.full(n_spatial, -1, dtype=np.intp)
    depth = np.full(n_spatial, np.inf)
    pseudo = np.full(n_spatial, float(fill_value))
    for i, p in enumerate(particles):
        sl = _window(p, x_c, halo)
        if sl is None:
            continue
        widx = _window_flat(sl, spatial_shape, strides).ravel()
        lv = np.asarray(p.level(_window_coords(sl, x_c)), dtype=float).ravel()
        np.minimum.at(pseudo, widx, lv)
        better = lv < np.minimum(depth[widx], 0.0)
        bidx = widx[better]
        owner[bidx] = i
        depth[bidx] = lv[better]
    solid_flat = owner >= 0
    pseudo = np.where(solid_flat, depth, pseudo)

    # --- face scan ----------------------------------------------------------
    solid_grid = solid_flat.reshape(spatial_shape)
    owner_grid = owner.reshape(spatial_shape)

    keys, thetas = [], []                       # directed theta registry
    contact_cols = {k: [] for k in ("lo", "hi", "i", "j", "s_i", "s_j")}
    for a in range(ndim):
        nb_solid, valid = _neighbor(solid_grid, a, +1)
        nb_owner, _ = _neighbor(owner_grid, a, +1)
        lo = np.flatnonzero((valid & (solid_grid != nb_solid)).ravel())
        if lo.size:
            hi = lo + strides[a]
            # fluid–solid crossing faces: owner of the solid cell intersects
            lo_solid = solid_flat[lo]
            own = np.where(lo_solid, owner[lo], owner[hi])
            p_lo = _cell_coords(lo, x_c, spatial_shape)
            p_hi = _cell_coords(hi, x_c, spatial_shape)
            s = np.empty(lo.size)
            for i in np.unique(own):
                m = own == i
                s[m] = particles[i].intersect(p_lo[m], p_hi[m])
            keys.append(_FaceThetaMap.pack(lo, a, +1, ndim))
            thetas.append(s)
            keys.append(_FaceThetaMap.pack(hi, a, -1, ndim))
            thetas.append(1.0 - s)
        # solid–solid faces between different particles
        cc = np.flatnonzero((valid & solid_grid & nb_solid
                             & (owner_grid != nb_owner)).ravel())
        if cc.size:
            hi = cc + strides[a]
            i_arr, j_arr = owner[cc], owner[hi]
            p_lo = _cell_coords(cc, x_c, spatial_shape)
            p_hi = _cell_coords(hi, x_c, spatial_shape)
            # gap test: cell strictly outside the *other* particle
            out_j = np.empty(cc.size, dtype=bool)   # level_j(p_lo) > 0
            out_i = np.empty(cc.size, dtype=bool)   # level_i(p_hi) > 0
            s_i = np.full(cc.size, np.nan)
            s_j = np.full(cc.size, np.nan)
            for i in np.unique(np.concatenate([i_arr, j_arr])):
                m = j_arr == i
                if m.any():
                    out_j[m] = particles[i].level(p_lo[m]) > 0.0
                m = i_arr == i
                if m.any():
                    out_i[m] = particles[i].level(p_hi[m]) > 0.0
            gap = out_i & out_j
            if gap.any():
                for i in np.unique(i_arr[gap]):
                    m = gap & (i_arr == i)
                    s_i[m] = particles[i].intersect(p_lo[m], p_hi[m])
                for j in np.unique(j_arr[gap]):
                    m = gap & (j_arr == j)
                    s_j[m] = particles[j].intersect(p_lo[m], p_hi[m])
                gap &= s_i < s_j            # disjoint surface intervals
            if gap.any():
                if contact == "error":
                    raise RuntimeError(
                        f"particle IBM: {int(gap.sum())} contact face(s) "
                        "between different particles (contact='error')")
                if contact == "no_flux":
                    contact_cols["lo"].append(cc[gap])
                    contact_cols["hi"].append(hi[gap])
                    contact_cols["i"].append(i_arr[gap])
                    contact_cols["j"].append(j_arr[gap])
                    contact_cols["s_i"].append(s_i[gap])
                    contact_cols["s_j"].append(s_j[gap])
                # 'merge': nothing to do

    if keys:
        theta_map = _FaceThetaMap(np.concatenate(keys), np.concatenate(thetas),
                                  ndim)
    else:
        theta_map = _FaceThetaMap(np.empty(0, dtype=np.intp), np.empty(0), ndim)

    # --- fluid–solid crossings through the shared IBM core ------------------
    out_s, in_s, meta = _construct_ibm_core(
        solid_grid, theta_map, x_c, axes, shape, rescale, pair=False)

    # --- contact crossings (no_flux policy) ---------------------------------
    n_contact = 0
    if contact_cols["lo"]:
        lo = np.concatenate(contact_cols["lo"])
        hi = np.concatenate(contact_cols["hi"])
        part_i = np.concatenate(contact_cols["i"])
        part_j = np.concatenate(contact_cols["j"])
        s_i = np.concatenate(contact_cols["s_i"])
        s_j = np.concatenate(contact_cols["s_j"])
        n_contact = lo.size
        ax_of = np.empty(lo.size, dtype=np.intp)
        for a in range(ndim):
            ax_of[(hi - lo) == strides[a]] = a
        out_s = _append_contact_side(out_s, lo, hi, ax_of, +1, s_i,
                                     0.5 * (s_i + s_j), owner, x_c,
                                     spatial_shape, strides, ndim, rescale)
        in_s = _append_contact_side(in_s, hi, lo, ax_of, -1, 1.0 - s_j,
                                    0.5 * (s_i + s_j), owner, x_c,
                                    spatial_shape, strides, ndim, rescale)

    ibm = _pair_sides_to_ibm(out_s, in_s, **meta)

    # --- normals + per-crossing bookkeeping ---------------------------------
    crossing_particle = owner[ibm.row_in]
    # A regular crossing has a fluid out cell; a contact crossing's out cell
    # is solid (it belongs to the partner particle).
    contact_mask = solid_flat[ibm.row_out]
    contact_partner = np.where(contact_mask, owner[ibm.row_out], -1)

    # The in-side particle's outward normal serves both crossing types: it
    # points solid -> fluid on regular crossings, and towards the partner
    # (the out side) on contact crossings.
    normals = np.empty((ibm.n_crossings, ndim))
    for i in np.unique(crossing_particle):
        m = crossing_particle == i
        normals[m] = particles[i].normal(ibm.coords[m])

    labels = np.where(solid_flat, owner + 1, 0).reshape(spatial_shape)
    sizes = np.bincount(labels.ravel(),
                        minlength=len(particles) + 1)[1:].astype(np.intp)
    seg = Segmentation(labels=labels, n_segments=len(particles),
                       region="negative", connectivity=1, sizes=sizes)

    info = ParticleIBMInfo(
        owner=owner_grid, crossing_particle=crossing_particle,
        contact=contact_mask, contact_partner=contact_partner,
        normals=normals, pseudo_sdf=pseudo.reshape(spatial_shape),
        segmentation=seg)
    return ibm, info


def _append_contact_side(side, cells, ghosts, ax_of, direction, theta_own,
                         theta_mid, owner, x_c, spatial_shape, strides, ndim,
                         rescale):
    """Append contact-crossing entries to one :class:`_IBMSide`.

    ``theta_own`` anchors the Lagrange reconstruction at this side's own
    particle surface; ``theta_mid`` (the mid-gap point measured from the
    *lo* cell) defines the shared crossing coordinates.
    """
    multi = np.unravel_index(cells, spatial_shape)
    n = cells.size
    coef_c = np.zeros(n)
    coef_o = np.zeros(n)
    coef_w = np.zeros(n)
    opp = np.full(n, -1, dtype=np.intp)

    xc = np.empty(n); xg = np.empty(n); xw = np.empty(n)
    xo = np.empty(n); opp_ok = np.zeros(n, dtype=bool)
    for a in range(ndim):
        m = ax_of == a
        if not m.any():
            continue
        ia = multi[a][m]
        xc[m] = x_c[a][ia]
        xg[m] = x_c[a][ia + direction]
        xw[m] = xc[m] + theta_own[m] * (xg[m] - xc[m])
        io = ia - direction
        ok = (io >= 0) & (io < spatial_shape[a])
        cand = cells[m] - direction * strides[a]
        same = np.zeros(ia.size, dtype=bool)
        same[ok] = owner[cand[ok]] == owner[cells[m][ok]]
        opp_ok[m] = same
        xo[m] = x_c[a][np.clip(io, 0, spatial_shape[a] - 1)]

    nrm = opp_ok
    if np.any(nrm):
        l0, l1, l2 = _lagrange3(xo[nrm], xc[nrm], xw[nrm], xg[nrm])
        coef_o[nrm] = l0
        coef_c[nrm] = l1
        coef_w[nrm] = l2
        opp[nrm] = cells[nrm] - direction * strides[ax_of[nrm]]
    fb = ~nrm
    if np.any(fb):
        coef_c[fb] = (xg[fb] - xw[fb]) / (xc[fb] - xw[fb])
        coef_w[fb] = (xg[fb] - xc[fb]) / (xw[fb] - xc[fb])

    lower = np.minimum(cells, ghosts)
    key = lower * ndim + ax_of

    coords = np.column_stack([x_c[a][multi[a]] for a in range(ndim)])
    # shared wall coordinate at the mid-gap point (measured from the lo cell)
    t_from_cell = theta_mid if direction == +1 else 1.0 - theta_mid
    for a in range(ndim):
        m = ax_of == a
        coords[m, a] = xc[m] + t_from_cell[m] * (xg[m] - xc[m])

    row = np.concatenate([side.row, cells.astype(np.intp)])
    coef_c_all = np.concatenate([side.coef_c, coef_c])
    coef_o_all = np.concatenate([side.coef_o, coef_o])
    coef_w_all = np.concatenate([side.coef_w_self, coef_w])
    coef_ws_all = np.concatenate([side.coef_w_sib, np.zeros(n)])

    row_scale = np.ones(side.n_cells)
    if rescale and row.size:
        max_coef = np.maximum.reduce(
            [np.abs(coef_c_all), np.abs(coef_o_all),
             np.abs(coef_w_all), np.abs(coef_ws_all)])
        per_cell = np.zeros(side.n_cells)
        np.maximum.at(per_cell, row, max_coef)
        cut = per_cell > 1.0
        row_scale[cut] = 1.0 / per_cell[cut]

    return _IBMSide(
        row=row,
        ghost=np.concatenate([side.ghost, ghosts.astype(np.intp)]),
        opp=np.concatenate([side.opp, opp]),
        coef_c=coef_c_all, coef_o=coef_o_all,
        coef_w_self=coef_w_all, coef_w_sib=coef_ws_all,
        sib=np.concatenate([side.sib, np.full(n, -1, dtype=np.intp)]),
        axis=np.concatenate([side.axis, ax_of]),
        direction=np.concatenate([side.direction,
                                  np.full(n, direction, dtype=np.intp)]),
        crossing_key=np.concatenate([side.crossing_key, key.astype(np.intp)]),
        coords=np.concatenate([side.coords, coords], axis=0),
        row_scale=row_scale, n_cells=side.n_cells,
    )


# ---------------------------------------------------------------------------
# Interface-condition sugar
# ---------------------------------------------------------------------------

def contact_conditions(base_ic, ibm, info, *, contact_ic=None):
    """Per-crossing ic: *base_ic* everywhere, a contact condition on contacts.

    Parameters
    ----------
    base_ic : tuple of two dicts
        Interface condition for the regular (fluid–solid) crossings, in the
        format of :func:`pymrm.apply_ibm_interface`.  Coefficients may be
        scalars or ns-broadcastable arrays (no per-crossing arrays).
    ibm : IBM
    info : ParticleIBMInfo
    contact_ic : tuple of two dicts, optional
        Condition imposed on the contact crossings.  Default: independent
        homogeneous Neumann on both sides (``q_out = 0`` and ``q_in = 0``) —
        no transport between the particles.

    Returns
    -------
    tuple of two dicts with per-crossing coefficient arrays.
    """
    if contact_ic is None:
        contact_ic = ({"a": (1.0, 0.0), "b": (0.0, 0.0), "d": 0.0},
                      {"a": (0.0, 1.0), "b": (0.0, 0.0), "d": 0.0})
    if not np.any(info.contact):
        return base_ic

    npnt = ibm.n_crossings
    m = info.contact.astype(float)              # (npnt,): 1 on contacts

    def blend_values(vb, vc):
        """Per-crossing mix of two scalar/ns-broadcastable coefficients."""
        vb = np.asarray(vb, dtype=float)
        vc = np.asarray(vc, dtype=float)
        trailing = np.broadcast_shapes(vb.shape, vc.shape)
        m_e = m.reshape((npnt,) + (1,) * len(trailing))
        return (np.broadcast_to(vb, trailing) * (1.0 - m_e)
                + np.broadcast_to(vc, trailing) * m_e)

    def blend(eq_base, eq_contact):
        out = {}
        for key in ("a", "b"):
            base_pair = (eq_base or {}).get(key, (0.0, 0.0))
            cont_pair = (eq_contact or {}).get(key, (0.0, 0.0))
            out[key] = tuple(blend_values(base_pair[s], cont_pair[s])
                             for s in range(2))
        out["d"] = blend_values((eq_base or {}).get("d", 0.0),
                                (eq_contact or {}).get("d", 0.0))
        return out

    return (blend(base_ic[0], contact_ic[0]), blend(base_ic[1], contact_ic[1]))
