"""One-sided normal-derivative operators for the :mod:`pymrm` immersed boundary method.

This module provides *Layer 1* of the generalized immersed-interface coupling:
for every wall crossing of an :class:`pymrm.ibm.IBM` object it constructs, on
each side separately, a linear formula for the outward normal derivative at the
interface point,

.. math::
    q_\\text{side} \\;=\\; \\left.\\frac{\\partial c}{\\partial n_\\text{side}}
    \\right|_\\Gamma \\;\\approx\\;
    \\alpha_\\text{side}\\, c_\\Gamma^\\text{side}
    \\;+\\; \\sum_{j \\in S_\\text{side}} \\gamma_j\\, c_j ,

where :math:`c_\\Gamma^\\text{side}` is the (generally unknown) one-sided
interface value and :math:`S_\\text{side}` contains nearby *same-side* cell
centres.  The weights are polynomially exact generalized finite-difference
(GFD) weights obtained from a minimum-weighted-norm problem.  These operators
are the building blocks for general interface conditions (conjugate diffusion,
partition coefficients, contact resistance, surface reactions) assembled in
:mod:`pymrm.ibm_coupling`.

Sign conventions
----------------
* The stored unit normal is ``n = grad(sdf)/|grad(sdf)|`` and points from the
  solid (``sdf < 0``) into the fluid (``sdf >= 0``).
* Derivatives are **outward per side**, matching
  :func:`pymrm.coupling.construct_interface_matrices`:
  ``q_out`` is taken along ``-n`` (out of the fluid domain) and ``q_in`` along
  ``+n`` (out of the solid domain).  For a healthy reconstruction both
  ``alpha_out`` and ``alpha_in`` are positive.

Robustness / length-scale strategy
----------------------------------
Stencil selection is governed by two caps: a geometric one
(``radius_factor * rings * h_local``) and a physical one
(``length_scale_factor * length_scale``).  When too few good points are
available inside the cap the polynomial degree is *lowered* instead of
reaching farther: ``p=2`` -> enlarge once -> ``p=1`` -> two-point normal
formula -> flagged unresolved.  Candidate cells must be flood-fill connected
to the cut cell through same-side cells, so points across a thin gap or a
neighbouring solid are never used.
"""

from dataclasses import dataclass
import math
import warnings
import numpy as np
from scipy.ndimage import label as _cc_label
from scipy.sparse import csr_array, coo_array
from scipy.spatial import cKDTree

from pymrm.ibm import _spatial_contributions, _ns_contributions, _is_pure_spatial

__all__ = [
    "IBMNormalDerivative",
    "construct_ibm_normal_derivative",
    "construct_ibm_normal_derivative_ops",
    "interface_normals",
    "gfd_normal_derivative_weights",
]

_DEGENERATE_PROJ = 1e-3   # |(x_cut - x_G) . u| < _DEGENERATE_PROJ * h -> axis fallback


# ---------------------------------------------------------------------------
# Public reconstruction container
# ---------------------------------------------------------------------------

@dataclass
class IBMNormalDerivative:
    """One-sided normal-derivative operators and diagnostics per IBM crossing.

    The outward normal derivative on each side of crossing ``k`` is

        ``q_side[k] = alpha_side[k] * c_gamma_side[k] + (D_side @ c_spatial)[k]``

    with ``c_spatial`` the field on the spatial grid (flattened, C-order) and
    the *outward per side* sign convention described in the module docstring.

    Attributes
    ----------
    normals : ndarray, shape (n_crossings, ndim_s)
        Unit normals, pointing from solid to fluid.
    alpha_out, alpha_in : ndarray, shape (n_crossings,)
        Weight of the one-sided interface value in ``q_side``.
    D_out, D_in : csr_array, shape (n_crossings, n_spatial_cells)
        Weights of the same-side cell values in ``q_side``.
    degree_out, degree_in : ndarray of int
        Polynomial degree actually used: 2 or 1 (GFD), 0 (two-point normal
        formula), -1 (degenerate axis-direction two-point fallback).
    n_stencil_out, n_stencil_in : ndarray of int
        Number of cell points in the stencil.
    radius_out, radius_in : ndarray
        Physical search radius of the accepted stencil.
    cond_out, cond_in : ndarray
        Conditioning of the GFD moment matrix (1 for two-point formulas).
    moment_residual_out, moment_residual_in : ndarray
        Max-norm residual of the polynomial moment conditions.
    weight_norm_out, weight_norm_in : ndarray
        ``h``-scaled 1-norm of the weights, O(1) for a healthy formula.
    unresolved_out, unresolved_in : ndarray of bool
        True where only the degenerate axis-direction fallback was possible.

    Shape information
    -----------------
    n_crossings : int
        Number of wall crossings (matches ``ibm.n_crossings``).
    spatial_shape : tuple
        Spatial grid shape.
    shape : tuple
        Full field shape (spatial + non-spatial axes).
    axes : tuple of int
        Which axes of ``shape`` are spatial.
    ns_size : int
        Product of the non-spatial dimensions (1 when purely spatial).
    n_cells : int
        Total number of cells, ``prod(shape)``.
    n_spatial_cells : int
        Number of spatial cells, ``prod(spatial_shape)``.
    h_ref : ndarray, shape (ndim_spatial,)
        Median cell-centre spacing along each spatial axis (reference scale
        for the GFD weights).
    """

    normals: np.ndarray

    alpha_out: np.ndarray
    alpha_in: np.ndarray
    D_out: csr_array
    D_in: csr_array

    degree_out: np.ndarray
    degree_in: np.ndarray
    n_stencil_out: np.ndarray
    n_stencil_in: np.ndarray
    radius_out: np.ndarray
    radius_in: np.ndarray
    cond_out: np.ndarray
    cond_in: np.ndarray
    moment_residual_out: np.ndarray
    moment_residual_in: np.ndarray
    weight_norm_out: np.ndarray
    weight_norm_in: np.ndarray
    unresolved_out: np.ndarray
    unresolved_in: np.ndarray

    n_crossings: int
    spatial_shape: tuple
    shape: tuple
    axes: tuple
    ns_size: int
    n_cells: int
    n_spatial_cells: int
    h_ref: np.ndarray


# ---------------------------------------------------------------------------
# GFD weight kernel
# ---------------------------------------------------------------------------

def _monomial_powers(dim, degree):
    """Multi-indices of all monomials up to *degree*, ordered by total degree."""
    powers = []

    def rec(current, remaining_dim, remaining_degree):
        if remaining_dim == 1:
            powers.append(current + [remaining_degree])
            return
        for k in range(remaining_degree + 1):
            rec(current + [k], remaining_dim - 1, remaining_degree - k)

    for total in range(degree + 1):
        rec([], dim, total)
    return np.array(powers, dtype=np.intp)


def _n_monomials(dim, degree):
    return math.comb(dim + degree, degree)


def _eval_monomials(Y, powers):
    """Evaluate monomials at nodes.  ``Y``: (..., dim) -> (..., M)."""
    P = np.ones(Y.shape[:-1] + (len(powers),))
    for m, alpha in enumerate(powers):
        for j, p in enumerate(alpha):
            if p:
                P[..., m] *= Y[..., j] ** p
    return P


def _gfd_weights_batched(Y, qinv, u_scaled, degree):
    """Batched minimum-weighted-norm GFD weights for the directional derivative.

    Parameters
    ----------
    Y : ndarray, shape (nb, N1, dim)
        Node coordinates scaled per crossing; node 0 is the interface point
        (a row of zeros).  Padded nodes are marked by ``qinv == 0``.
    qinv : ndarray, shape (nb, N1)
        Inverse minimum-norm weights; exactly zero for padded nodes.
    u_scaled : ndarray, shape (nb, dim)
        Derivative direction divided by the per-crossing scaling ``hvec`` (so
        the weights of the *scaled* problem approximate the *physical*
        directional derivative).
    degree : int
        Polynomial exactness degree.

    Returns
    -------
    a : ndarray, shape (nb, N1)
        Weights (node 0 = interface value).  Rows of failed solves are valid
        numbers but must be discarded based on *cond*/*moment_res*.
    cond : ndarray, shape (nb,)
        Condition number of the moment matrix ``P^T Q^-1 P``.
    moment_res : ndarray, shape (nb,)
        Max-norm of ``P^T a - d``.
    """
    nb, _, dim = Y.shape
    powers = _monomial_powers(dim, degree)
    M = len(powers)

    P = _eval_monomials(Y, powers)                       # (nb, N1, M)
    d = np.zeros((nb, M))
    for m, alpha in enumerate(powers):
        if alpha.sum() == 1:
            j = int(np.argmax(alpha == 1))
            d[:, m] = u_scaled[:, j]

    A = np.einsum("kim,ki,kil->kml", P, qinv, P)         # (nb, M, M)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        cond = np.linalg.cond(A)
    cond = np.where(np.isfinite(cond), cond, np.inf)

    good = cond < 1.0 / np.finfo(float).eps
    A_safe = np.where(good[:, None, None], A, np.eye(M)[None, :, :])
    lam = np.linalg.solve(A_safe, d[:, :, np.newaxis])[:, :, 0]
    a = qinv * np.einsum("kim,km->ki", P, lam)

    moment_res = np.abs(np.einsum("kim,ki->km", P, a) - d).max(axis=1)
    moment_res = np.where(good, moment_res, np.inf)
    return a, cond, moment_res


def gfd_normal_derivative_weights(x_gamma, x_cells, direction, degree=2,
                                  hvec=None, weight_power=4,
                                  interface_penalty=1.0):
    """GFD weights for a directional derivative at a single interface point.

    Returns ``alpha``, ``gamma`` and diagnostics such that

        ``dc/du|_Gamma ~= alpha * c_gamma + gamma @ c_cells``

    is exact for all polynomials up to *degree*.  This is the single-point
    convenience form of the batched kernel used by
    :func:`construct_ibm_normal_derivative`; *direction* is the (not necessarily
    unit) derivative direction ``u``.

    Parameters
    ----------
    x_gamma : array_like, shape (ndim,)
        Interface point.
    x_cells : array_like, shape (N, ndim)
        Same-side cell centres.
    direction : array_like, shape (ndim,)
        Derivative direction; normalized internally.
    degree : int, optional
        Polynomial exactness degree (default 2).
    hvec : array_like, shape (ndim,), optional
        Per-axis coordinate scaling.  Defaults to the median point distance.
    weight_power : float, optional
        Distance-penalty exponent: ``q_i = 1 + r_i**weight_power`` in scaled
        coordinates (default 4).
    interface_penalty : float, optional
        Minimum-norm weight of the interface node (default 1.0).

    Returns
    -------
    alpha : float
        Weight of the interface value.
    gamma : ndarray, shape (N,)
        Weights of the cell values.
    info : dict
        ``cond`` and ``moment_residual`` diagnostics.
    """
    x_gamma = np.asarray(x_gamma, dtype=float).ravel()
    x_cells = np.atleast_2d(np.asarray(x_cells, dtype=float))
    u = np.asarray(direction, dtype=float).ravel()
    u = u / np.linalg.norm(u)
    dim = x_gamma.size

    if hvec is None:
        r = np.linalg.norm(x_cells - x_gamma, axis=1)
        h = np.median(r[r > 0]) if np.any(r > 0) else 1.0
        hvec = np.full(dim, h)
    else:
        hvec = np.asarray(hvec, dtype=float).ravel()

    Y = np.vstack([np.zeros(dim), (x_cells - x_gamma) / hvec])[None, :, :]
    r = np.linalg.norm(Y[0], axis=1)
    q = 1.0 + r ** weight_power
    q[0] = interface_penalty
    qinv = (1.0 / q)[None, :]
    u_scaled = (u / hvec)[None, :]

    a, cond, moment_res = _gfd_weights_batched(Y, qinv, u_scaled, degree)
    info = {"cond": float(cond[0]), "moment_residual": float(moment_res[0])}
    return float(a[0, 0]), a[0, 1:], info


# ---------------------------------------------------------------------------
# Normals
# ---------------------------------------------------------------------------

def _normalize_x_c(x_c, ndim_s):
    if isinstance(x_c, np.ndarray) and x_c.ndim == 1:
        x_c = [x_c]
    else:
        x_c = [np.asarray(xci, dtype=float) for xci in x_c]
    if len(x_c) != ndim_s:
        raise ValueError(f"len(x_c)={len(x_c)} must equal sdf.ndim={ndim_s}")
    return x_c


def _axis_spacings(x_c):
    """Local cell-centre spacing per axis (length-1 axes get spacing 1)."""
    return [np.gradient(xc) if xc.size > 1 else np.ones(1) for xc in x_c]


def _axis_fallback_normals(ibm, mask=None):
    """Axis-aligned solid->fluid normals for the crossings in *mask*."""
    ndim_s = len(ibm.spatial_shape)
    idx = np.arange(ibm.n_crossings) if mask is None else np.flatnonzero(mask)
    n = np.zeros((idx.size, ndim_s))
    # direction points from the fluid cut cell to its solid ghost, so the
    # solid->fluid direction along the crossing axis is -direction.
    n[np.arange(idx.size), ibm.axis[idx]] = -ibm.direction[idx]
    return n


def interface_normals(ibm, sdf, x_c):
    """Unit interface normals (solid -> fluid) at each wall crossing.

    The cell-centred gradient of *sdf* is computed with :func:`numpy.gradient`
    (non-equidistant aware) and averaged over the two cut cells of each
    crossing.  Crossings with a degenerate gradient (kinks of the level set,
    e.g. near medial axes of thin solids) fall back to the axis-aligned
    normal of the crossing.

    Parameters
    ----------
    ibm : IBM
        Immersed-boundary data from :func:`pymrm.construct_ibm`.
    sdf : array_like
        Signed-distance (or level-set) field at the spatial cell centres.
    x_c : array_like or list of array_like
        Cell-centre coordinates, one 1-D array per spatial axis.

    Returns
    -------
    ndarray, shape (n_crossings, ndim_spatial)
    """
    sdf = np.asarray(sdf, dtype=float)
    ndim_s = sdf.ndim
    x_c = _normalize_x_c(x_c, ndim_s)

    if ibm.n_crossings == 0:
        return np.empty((0, ndim_s))

    if ndim_s == 1:
        grads = [np.gradient(sdf, x_c[0])]
    else:
        grads = np.gradient(sdf, *x_c)
    g = np.stack([gr.ravel() for gr in grads], axis=1)   # (n_spatial, ndim_s)

    # Interpolate the two cut-cell gradients to the wall position:
    # x_w = x_out + theta * (x_in - x_out) with theta from the sdf values.
    sdf_flat = sdf.ravel()
    sdf_o = sdf_flat[ibm.row_out]
    sdf_i = sdf_flat[ibm.row_in]
    denom = sdf_o - sdf_i
    theta = np.where(denom != 0.0, sdf_o / np.where(denom == 0.0, 1.0, denom),
                     0.5)
    theta = np.clip(theta, 0.0, 1.0)[:, np.newaxis]
    n = (1.0 - theta) * g[ibm.row_out] + theta * g[ibm.row_in]
    norm = np.linalg.norm(n, axis=1)
    med = np.median(norm)
    degen = norm <= max(1e-6 * med, 1e-300)
    if np.any(degen):
        n[degen] = _axis_fallback_normals(ibm, degen)
        norm = np.linalg.norm(n, axis=1)
    return n / norm[:, np.newaxis]


# ---------------------------------------------------------------------------
# Stencil selection and per-side reconstruction
# ---------------------------------------------------------------------------

def _flood_filter(cand, seed, spatial_shape, strides):
    """Keep the candidates flood-fill connected to *seed* within the set.

    Adjacency is face adjacency on the structured grid, restricted to the
    candidate set itself, so cells that are geometrically close but reachable
    only around a thin obstacle are rejected.
    """
    pos = {int(c): i for i, c in enumerate(cand)}
    multi = np.unravel_index(cand, spatial_shape)
    keep = np.zeros(cand.size, dtype=bool)
    stack = [int(seed)]
    keep[pos[int(seed)]] = True
    ndim_s = len(spatial_shape)
    while stack:
        c = stack.pop()
        i = pos[c]
        for a in range(ndim_s):
            ia = multi[a][i]
            if ia + 1 < spatial_shape[a]:
                nb = c + int(strides[a])
                j = pos.get(nb)
                if j is not None and not keep[j]:
                    keep[j] = True
                    stack.append(nb)
            if ia > 0:
                nb = c - int(strides[a])
                j = pos.get(nb)
                if j is not None and not keep[j]:
                    keep[j] = True
                    stack.append(nb)
    return cand[keep]


def _build_recon_side(ibm, region_flat, seeds, u_dir, x_c, hax, h_ref, *,
                      degree_target, length_scale, rings, radius_factor,
                      length_scale_factor, enlarge_factor, min_points_factor,
                      weight_power, interface_penalty, cond_max, moment_tol,
                      weight_norm_max, connectivity, side_name):
    """Select stencils and compute GFD weights for one side of the interface."""
    npnt = ibm.n_crossings
    spatial_shape = ibm.spatial_shape
    ndim_s = len(spatial_shape)
    strides = np.array(
        [math.prod(spatial_shape[a + 1:]) for a in range(ndim_s)], dtype=np.intp
    )
    x_gamma = ibm.coords                                  # (npnt, ndim_s)

    # --- same-side cells, connected components, KD-tree (physical coords) ---
    side_idx = np.flatnonzero(region_flat)
    side_multi = np.unravel_index(side_idx, spatial_shape)
    side_pts = np.column_stack([x_c[a][side_multi[a]] for a in range(ndim_s)])
    labels, _ = _cc_label(region_flat.reshape(spatial_shape))
    labels_flat = labels.ravel()
    tree = cKDTree(side_pts)

    # --- per-crossing local cell size and search radii ---
    seed_multi = np.unravel_index(seeds, spatial_shape)
    hvec = np.column_stack([hax[a][seed_multi[a]] for a in range(ndim_s)])
    h_char = hvec.max(axis=1)                             # (npnt,)
    cap1 = radius_factor * rings * h_char
    cap2 = length_scale_factor * length_scale             # (npnt,) possibly inf
    r0 = np.minimum(cap1, cap2)

    def gather(k, radius):
        """Filtered candidate spatial flat indices for crossing k."""
        local = tree.query_ball_point(x_gamma[k], radius)
        cand = side_idx[np.asarray(local, dtype=np.intp)]
        if seeds[k] not in cand:
            cand = np.append(cand, seeds[k])
        cand = cand[labels_flat[cand] == labels_flat[seeds[k]]]
        if connectivity == "flood":
            cand = _flood_filter(cand, seeds[k], spatial_shape, strides)
        elif connectivity == "halfspace":
            cm = np.unravel_index(cand, spatial_shape)
            pts = np.column_stack([x_c[a][cm[a]] for a in range(ndim_s)])
            proj = (pts - x_gamma[k]) @ u_dir[k]
            keep = proj <= 0.5 * h_char[k]
            keep[cand == seeds[k]] = True
            cand = cand[keep]
        elif connectivity != "none":
            raise ValueError(
                f"connectivity must be 'flood', 'halfspace' or 'none', "
                f"got {connectivity!r}"
            )
        return cand

    cand0 = [gather(k, r0[k]) for k in range(npnt)]

    # --- result arrays ---
    alpha = np.zeros(npnt)
    stencil = [None] * npnt
    weights = [None] * npnt
    degree_used = np.full(npnt, -2, dtype=np.intp)
    cond = np.zeros(npnt)
    moment_res = np.zeros(npnt)
    radius_used = r0.copy()
    unresolved = np.zeros(npnt, dtype=bool)

    def attempt(active, cands, radii, p_arr):
        """Batched GFD attempt; returns indices accepted (and records them)."""
        accepted = []
        for p in np.unique(p_arr[active]):
            if p < 1:
                continue
            M = _n_monomials(ndim_s, int(p))
            min_pts = math.ceil(min_points_factor * (M - 1))
            n_max = 4 * M
            group = [k for k in active
                     if p_arr[k] == p and cands[k].size >= min_pts]
            if not group:
                continue
            # Truncate to the n_max nearest candidates.
            trunc = []
            for k in group:
                c = cands[k]
                if c.size > n_max:
                    cm = np.unravel_index(c, spatial_shape)
                    pts = np.column_stack(
                        [x_c[a][cm[a]] for a in range(ndim_s)])
                    r = np.linalg.norm(pts - x_gamma[k], axis=1)
                    c = c[np.argsort(r, kind="stable")[:n_max]]
                trunc.append(c)
            n1 = 1 + max(c.size for c in trunc)
            nb = len(group)
            Y = np.zeros((nb, n1, ndim_s))
            qinv = np.zeros((nb, n1))
            u_scaled = np.empty((nb, ndim_s))
            for i, (k, c) in enumerate(zip(group, trunc)):
                cm = np.unravel_index(c, spatial_shape)
                pts = np.column_stack([x_c[a][cm[a]] for a in range(ndim_s)])
                Yk = (pts - x_gamma[k]) / hvec[k]
                Y[i, 1:1 + c.size] = Yk
                r = np.linalg.norm(Yk, axis=1)
                qinv[i, 1:1 + c.size] = 1.0 / (1.0 + r ** weight_power)
                qinv[i, 0] = 1.0 / interface_penalty
                u_scaled[i] = u_dir[k] / hvec[k]
            a, cnd, mres = _gfd_weights_batched(Y, qinv, u_scaled, int(p))
            wnorm = h_char[np.array(group)] * np.abs(a).sum(axis=1)
            ok = (cnd <= cond_max) & (mres <= moment_tol) & \
                 (wnorm <= weight_norm_max)
            for i, k in enumerate(group):
                if not ok[i]:
                    continue
                c = trunc[i]
                alpha[k] = a[i, 0]
                stencil[k] = c
                weights[k] = a[i, 1:1 + c.size].copy()
                degree_used[k] = p
                cond[k] = cnd[i]
                moment_res[k] = mres[i]
                radius_used[k] = radii[k]
                accepted.append(k)
        return accepted

    active = set(range(npnt))

    # Stage 1: target degree at the initial radius.
    p_target = degree_target.astype(np.intp)
    for k in attempt(sorted(active), cand0, r0, p_target):
        active.discard(k)

    # Stage 2: enlarge once (never beyond the length-scale cap), retry target.
    if active:
        r1 = np.minimum(enlarge_factor * r0, cap2)
        grow = [k for k in sorted(active)
                if p_target[k] >= 1 and r1[k] > r0[k] * (1 + 1e-12)]
        if grow:
            cand1 = list(cand0)
            for k in grow:
                cand1[k] = gather(k, r1[k])
            for k in attempt(grow, cand1, r1, p_target):
                active.discard(k)

    # Stage 3: demote the degree step by step at the original radius.
    p_cur = p_target.copy()
    while active and np.any(p_cur[sorted(active)] > 1):
        p_cur = np.maximum(p_cur - 1, 1)
        sub = [k for k in sorted(active) if p_target[k] >= 1]
        for k in attempt(sub, cand0, r0, p_cur):
            active.discard(k)

    # Final stage: two-point normal formula from the cut cell.
    n_axis_fallback = 0
    for k in sorted(active):
        x_cut = np.array([x_c[a][seed_multi[a][k]] for a in range(ndim_s)])
        s = float((x_cut - x_gamma[k]) @ u_dir[k])
        deg = 0
        if abs(s) < _DEGENERATE_PROJ * h_char[k]:
            u_ax = np.zeros(ndim_s)
            u_ax[ibm.axis[k]] = u_dir[k][ibm.axis[k]]
            nrm = np.linalg.norm(u_ax)
            if nrm == 0.0:
                u_ax[ibm.axis[k]] = 1.0
                nrm = 1.0
            s = float((x_cut - x_gamma[k]) @ (u_ax / nrm))
            deg = -1
            unresolved[k] = True
            n_axis_fallback += 1
        alpha[k] = -1.0 / s
        stencil[k] = np.array([seeds[k]], dtype=np.intp)
        weights[k] = np.array([1.0 / s])
        degree_used[k] = deg
        cond[k] = 1.0
        moment_res[k] = 0.0

    if n_axis_fallback:
        warnings.warn(
            f"IBM reconstruction ({side_name}): {n_axis_fallback} crossing(s) "
            "fell back to the degenerate axis-direction two-point formula "
            "(normal nearly tangent to the offset vector); flagged as "
            "unresolved.",
            RuntimeWarning, stacklevel=3,
        )

    # --- assemble sparse D and diagnostics ---
    if npnt:
        rows = np.concatenate(
            [np.full(stencil[k].size, k, dtype=np.intp) for k in range(npnt)])
        cols = np.concatenate(stencil)
        data = np.concatenate(weights)
    else:
        rows = cols = np.empty(0, dtype=np.intp)
        data = np.empty(0)
    D = coo_array((data, (rows, cols)),
                  shape=(npnt, ibm.n_spatial_cells)).tocsr()

    n_stencil = np.array([0 if stencil[k] is None else stencil[k].size
                          for k in range(npnt)], dtype=np.intp)
    weight_norm = h_char * (np.abs(alpha)
                            + np.array([np.abs(weights[k]).sum() if npnt else 0.0
                                        for k in range(npnt)]))
    return {
        "alpha": alpha, "D": D, "degree": degree_used, "n_stencil": n_stencil,
        "radius": radius_used, "cond": cond, "moment_residual": moment_res,
        "weight_norm": weight_norm, "unresolved": unresolved,
    }


def construct_ibm_normal_derivative(ibm, sdf, x_c, *, degree=2,
                                    length_scale=None, rings=2,
                                    radius_factor=1.0, length_scale_factor=0.5,
                                    enlarge_factor=1.5, min_points_factor=1.5,
                                    weight_power=4, interface_penalty=1.0,
                                    cond_max=1e8, moment_tol=1e-8,
                                    weight_norm_max=100.0, connectivity="flood",
                                    normals=None):
    """Construct one-sided normal-derivative operators for every IBM crossing.

    Parameters
    ----------
    ibm : IBM
        Immersed-boundary data from :func:`pymrm.construct_ibm` (or the
        particle front end :func:`pymrm.construct_ibm_particles`).
    sdf : array_like
        Cell-centred field whose sign classifies the regions (``sdf < 0``
        solid).  Usually the signed-distance field used to build *ibm*; with
        the particle front end pass ``ParticleIBMInfo.pseudo_sdf``.  When
        *normals* are supplied the field is used only for region
        classification and stencil selection, not differentiated for normals.
    x_c : array_like or list of array_like
        Cell-centre coordinates, one 1-D array per spatial axis.
    degree : int or ndarray of int, optional
        Target polynomial degree (default 2).  A per-crossing array caps the
        degree individually (hook for solution-adaptive order control).
        ``0`` forces the two-point normal formula.
    length_scale : None, float or ndarray, optional
        Relevant physical length scale ``L`` of the fields near the interface
        (particle size, boundary-layer thickness, ...).  Stencil radii never
        exceed ``length_scale_factor * L``; when too few points fit under the
        cap the degree is lowered instead of reaching farther.  ``None``
        (default) means the grid is assumed to resolve all relevant scales.
    rings : int, optional
        Nominal stencil extent in cells; the geometric radius cap is
        ``radius_factor * rings * h_local``.
    radius_factor, length_scale_factor, enlarge_factor : float, optional
        Radius-cap tuning; see above.  On rejection the radius is enlarged
        once by *enlarge_factor* (never beyond the length-scale cap) before
        the degree is lowered.
    min_points_factor : float, optional
        Accept degree ``p`` (``M_p`` monomials) only with at least
        ``ceil(min_points_factor * (M_p - 1))`` stencil points.
    weight_power, interface_penalty : float, optional
        Minimum-norm weights ``q_i = 1 + r_i**weight_power`` (scaled
        coordinates) and the interface-node weight.
    cond_max, moment_tol, weight_norm_max : float, optional
        Acceptance thresholds on the moment-matrix condition number, the
        moment residual and the ``h``-scaled 1-norm of the weights.
    connectivity : {'flood', 'halfspace', 'none'}, optional
        Same-side visibility filter.  ``'flood'`` (default, most robust)
        keeps only candidates flood-fill connected to the cut cell within the
        candidate set; ``'halfspace'`` uses a cheap normal half-space test.
    normals : ndarray, shape (n_crossings, ndim_s), optional
        Override the SDF-gradient normals (e.g. analytic normals).

    Returns
    -------
    IBMNormalDerivative
    """
    sdf = np.asarray(sdf, dtype=float)
    if sdf.shape != ibm.spatial_shape:
        raise ValueError(
            f"sdf.shape={sdf.shape} != ibm.spatial_shape={ibm.spatial_shape}")
    ndim_s = sdf.ndim
    x_c = _normalize_x_c(x_c, ndim_s)
    npnt = ibm.n_crossings

    if normals is None:
        normals = interface_normals(ibm, sdf, x_c)
    else:
        normals = np.asarray(normals, dtype=float)
        if normals.shape != (npnt, ndim_s):
            raise ValueError(
                f"normals shape {normals.shape} != ({npnt}, {ndim_s})")
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    degree_arr = np.broadcast_to(
        np.asarray(degree, dtype=np.intp), (npnt,)).copy()
    if length_scale is None:
        ls = np.full(npnt, np.inf)
    else:
        ls = np.broadcast_to(
            np.asarray(length_scale, dtype=float), (npnt,)).copy()

    hax = _axis_spacings(x_c)
    h_ref = np.array([np.median(h) for h in hax])

    region_fluid = sdf.ravel() >= 0.0
    common = dict(
        degree_target=degree_arr, length_scale=ls, rings=rings,
        radius_factor=radius_factor, length_scale_factor=length_scale_factor,
        enlarge_factor=enlarge_factor, min_points_factor=min_points_factor,
        weight_power=weight_power, interface_penalty=interface_penalty,
        cond_max=cond_max, moment_tol=moment_tol,
        weight_norm_max=weight_norm_max, connectivity=connectivity,
    )
    # Outward per side: q_out along -n (out of the fluid), q_in along +n.
    out = _build_recon_side(ibm, region_fluid, ibm.row_out, -normals,
                            x_c, hax, h_ref, side_name="outside", **common)
    inn = _build_recon_side(ibm, ~region_fluid, ibm.row_in, +normals,
                            x_c, hax, h_ref, side_name="inside", **common)

    return IBMNormalDerivative(
        normals=normals,
        alpha_out=out["alpha"], alpha_in=inn["alpha"],
        D_out=out["D"], D_in=inn["D"],
        degree_out=out["degree"], degree_in=inn["degree"],
        n_stencil_out=out["n_stencil"], n_stencil_in=inn["n_stencil"],
        radius_out=out["radius"], radius_in=inn["radius"],
        cond_out=out["cond"], cond_in=inn["cond"],
        moment_residual_out=out["moment_residual"],
        moment_residual_in=inn["moment_residual"],
        weight_norm_out=out["weight_norm"], weight_norm_in=inn["weight_norm"],
        unresolved_out=out["unresolved"], unresolved_in=inn["unresolved"],
        n_crossings=npnt,
        spatial_shape=ibm.spatial_shape, shape=ibm.shape, axes=ibm.axes,
        ns_size=ibm.ns_size, n_cells=ibm.n_cells,
        n_spatial_cells=ibm.n_spatial_cells, h_ref=h_ref,
    )


# ---------------------------------------------------------------------------
# Non-spatial (component) expansion
# ---------------------------------------------------------------------------

def construct_ibm_normal_derivative_ops(ibm, recon):
    """Expand the reconstruction operators to the full field layout.

    Returns operators acting on the *flattened full field* ``c`` (including
    non-spatial axes) with rows ordered ``k * ns_size + j`` for crossing ``k``
    and non-spatial layer ``j`` — the same column ordering used by the
    ``G_out``/``G_in`` source matrices of :func:`pymrm.apply_ibm`:

        ``q_side.ravel() = alpha_side_full * w_side + N_side @ c.ravel()``

    Parameters
    ----------
    ibm : IBM
    recon : IBMNormalDerivative

    Returns
    -------
    alpha_out_full : ndarray, shape (n_crossings * ns_size,)
    N_out : csr_array, shape (n_crossings * ns_size, n_cells)
    alpha_in_full : ndarray, shape (n_crossings * ns_size,)
    N_in : csr_array, shape (n_crossings * ns_size, n_cells)
    """
    ns = ibm.ns_size
    npnt = ibm.n_crossings

    def expand(D):
        if _is_pure_spatial(ibm):
            return csr_array(D.copy())
        coo = D.tocoo()
        sc = _spatial_contributions(coo.col, ibm.shape, ibm.axes)
        ns_c = _ns_contributions(ibm.shape, ibm.axes)
        rows = (coo.row.astype(np.intp) * ns)[:, np.newaxis] \
            + np.arange(ns, dtype=np.intp)[np.newaxis, :]
        cols = sc[:, np.newaxis] + ns_c[np.newaxis, :]
        data = np.broadcast_to(coo.data[:, np.newaxis], rows.shape)
        return coo_array(
            (data.ravel(), (rows.ravel(), cols.ravel())),
            shape=(npnt * ns, ibm.n_cells),
        ).tocsr()

    alpha_out_full = np.repeat(recon.alpha_out, ns)
    alpha_in_full = np.repeat(recon.alpha_in, ns)
    return alpha_out_full, expand(recon.D_out), alpha_in_full, expand(recon.D_in)
