"""Tests for particle-based immersed boundaries (:mod:`pymrm.particles`)."""

import math

import numpy as np
import pytest
from scipy.sparse import csc_array
from scipy.sparse.linalg import spsolve

from pymrm.ibm import construct_ibm, apply_ibm, apply_ibm_vector
from pymrm.ibm_recon import construct_ibm_normal_derivative
from pymrm.ibm_coupling import construct_ibm_boundary_values
from pymrm.operators import construct_grad, construct_div
from pymrm.particles import (
    Sphere, Circle, Box, AnalyticParticle, GridParticle,
    construct_ibm_particles, contact_conditions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid(N, lo=0.0, hi=1.0):
    x_f = np.linspace(lo, hi, N + 1)
    x_c = 0.5 * (x_f[1:] + x_f[:-1])
    return x_f, x_c


def _grid2(N):
    x_f, x_c = _grid(N)
    X, Y = np.meshgrid(x_c, x_c, indexing="ij")
    return x_f, [x_c, x_c], X, Y


def _touching_spheres(gap=0.0, r=0.22):
    d = r + gap / 2.0
    return [Sphere((0.5 - d, 0.5015), r), Sphere((0.5 + d, 0.5015), r)]


def _union_sdf(particles, X, Y):
    pts = np.stack([X, Y], axis=-1)
    return np.minimum.reduce([p.level(pts) for p in particles])


# ---------------------------------------------------------------------------
# Particle shapes
# ---------------------------------------------------------------------------

def test_sphere_geometry():
    s = Sphere((0.3, -0.2), 0.5)
    rng = np.random.default_rng(0)
    # points inside and outside straddle the surface
    ang = rng.uniform(0, 2 * np.pi, 32)
    p_in = s.position + 0.3 * np.column_stack([np.cos(ang), np.sin(ang)])
    p_out = s.position + 0.9 * np.column_stack([np.cos(ang), np.sin(ang)])
    assert np.all(s.level(p_in) < 0) and np.all(s.level(p_out) > 0)
    t = s.intersect(p_out, p_in)
    xw = p_out + t[:, None] * (p_in - p_out)
    np.testing.assert_allclose(np.linalg.norm(xw - s.position, axis=1), 0.5,
                               atol=1e-14)
    n = s.normal(xw)
    np.testing.assert_allclose(n, (xw - s.position) / 0.5, atol=1e-14)
    # exiting segments (p0 inside) pick the other root
    t2 = s.intersect(p_in, p_out)
    xw2 = p_in + t2[:, None] * (p_out - p_in)
    np.testing.assert_allclose(np.linalg.norm(xw2 - s.position, axis=1), 0.5,
                               atol=1e-14)


def test_circle_alias():
    assert Circle is Sphere


def test_box_rotation():
    b = Box((1.0, 1.0), (0.4, 0.2), orientation=math.pi / 4)
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    # body corner (0.4, 0.2) in world coordinates
    corner = np.array([1.0 + 0.4 * c - 0.2 * s, 1.0 + 0.4 * s + 0.2 * c])
    assert abs(b.level(corner)) < 1e-12
    assert b.level(np.array([1.0, 1.0])) < 0          # centre inside
    (xlo, xhi), (ylo, yhi) = b.bounding_box()
    assert xlo <= corner[0] <= xhi and ylo <= corner[1] <= yhi
    # normal at the middle of the long face, rotated
    face_pt_body = np.array([[0.0, 0.2]])
    face_pt = b.position + face_pt_body @ np.array([[c, s], [-s, c]])
    n = b.normal(face_pt)
    np.testing.assert_allclose(n, [[-s, c]], atol=1e-12)


def test_analytic_particle_ellipse():
    ab = np.array([0.4, 0.25])
    ell = AnalyticParticle(
        lambda x: np.linalg.norm(x / ab, axis=-1) - 1.0,
        bounding_box=((-0.4, 0.4), (-0.25, 0.25)),
        position=(0.5, 0.5))
    pt = np.array([[0.5 + 0.4, 0.5]])
    assert abs(ell.level(pt)) < 1e-12
    n = ell.normal(pt)                     # numeric-FD default
    np.testing.assert_allclose(n, [[1.0, 0.0]], atol=1e-6)


def test_grid_particle_matches_sphere():
    # sample a sphere level function on a local grid, then rotate it
    xl = np.linspace(-0.3, 0.3, 61)
    Xl, Yl = np.meshgrid(xl, xl, indexing="ij")
    values = np.hypot(Xl, Yl) - 0.2
    gp = GridParticle(values, [xl, xl], position=(0.5, 0.5),
                      orientation=0.3)
    ref = Sphere((0.5, 0.5), 0.2)
    rng = np.random.default_rng(1)
    pts = rng.uniform(0.3, 0.7, (64, 2))
    np.testing.assert_allclose(gp.level(pts), ref.level(pts), atol=5e-5)
    ang = rng.uniform(0, 2 * np.pi, 16)
    surf = ref.position + 0.2 * np.column_stack([np.cos(ang), np.sin(ang)])
    np.testing.assert_allclose(gp.normal(surf), ref.normal(surf), atol=1e-3)
    # outside the local grid the sign stays positive
    assert gp.level(np.array([[5.0, 5.0]])) > 1.0


# ---------------------------------------------------------------------------
# Assembly: equivalence with the SDF path
# ---------------------------------------------------------------------------

def test_single_sphere_matches_sdf_path():
    N = 48
    x_f, x_c, X, Y = _grid2(N)
    s = Sphere((0.503, 0.497), 0.28)
    sdf = _union_sdf([s], X, Y)

    ibm_ref = construct_ibm(sdf, x_c)
    ibm_p, info = construct_ibm_particles([s], x_c)

    assert ibm_p.n_crossings == ibm_ref.n_crossings
    np.testing.assert_array_equal(ibm_p.crossing_key, ibm_ref.crossing_key)
    np.testing.assert_array_equal(ibm_p.row_out, ibm_ref.row_out)
    np.testing.assert_array_equal(ibm_p.row_in, ibm_ref.row_in)
    # exact wall positions: crossings lie exactly on the sphere surface
    r = np.linalg.norm(ibm_p.coords - s.position, axis=1)
    np.testing.assert_allclose(r, 0.28, atol=1e-13)
    # exact normals
    nex = (ibm_p.coords - s.position) / r[:, None]
    np.testing.assert_allclose(info.normals, nex, atol=1e-13)
    # classification bookkeeping
    assert not info.contact.any()
    assert np.array_equal(info.owner >= 0, sdf < 0)
    assert np.array_equal(info.pseudo_sdf < 0, sdf < 0)
    assert np.all(info.crossing_particle == 0)


def test_ns_axes_supported():
    N, Nc = 24, 3
    x_f, x_c, X, Y = _grid2(N)
    s = Sphere((0.5, 0.5), 0.25)
    ibm, info = construct_ibm_particles([s], x_c, axes=(0, 1),
                                        shape=(N, N, Nc))
    assert ibm.ns_size == Nc
    assert ibm.n_cells == N * N * Nc


def test_boundary_adjacent_particle():
    """A particle sticking out of the domain: clipped windows still work."""
    N = 32
    x_f, x_c, X, Y = _grid2(N)
    s = Sphere((0.0, 0.5), 0.2)          # half outside the domain
    ibm, info = construct_ibm_particles([s], x_c)
    assert ibm.n_crossings > 0
    assert np.all(np.isfinite(info.normals))
    # classification matches direct level evaluation on the full grid
    lv = s.level(np.stack([X, Y], axis=-1))
    assert np.array_equal(info.owner >= 0, lv < 0)


def test_sphere_3d_smoke():
    N = 16
    x_f, x_c1 = _grid(N)
    x_c = [x_c1, x_c1, x_c1]
    s = Sphere((0.5, 0.5, 0.5), 0.3)
    ibm, info = construct_ibm_particles([s], x_c)
    assert ibm.n_crossings > 0
    r = np.linalg.norm(ibm.coords - s.position, axis=1)
    np.testing.assert_allclose(r, 0.3, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(info.normals, axis=1), 1.0,
                               atol=1e-12)


# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------

def test_contact_regimes():
    N = 64
    x_f, x_c, X, Y = _grid2(N)
    h = 1.0 / N

    # resolved gap: no contacts, matches the SDF path
    parts = _touching_spheres(gap=3 * h)
    sdf = _union_sdf(parts, X, Y)
    ibm, info = construct_ibm_particles(parts, x_c)
    assert not info.contact.any()
    np.testing.assert_array_equal(ibm.crossing_key,
                                  construct_ibm(sdf, x_c).crossing_key)

    # sub-h gap and exact touch: contacts appear under no_flux
    for gap in (0.3 * h, 0.0):
        parts = _touching_spheres(gap=gap)
        ibm, info = construct_ibm_particles(parts, x_c)
        assert info.contact.any()
        pairs = np.stack([info.crossing_particle[info.contact],
                          info.contact_partner[info.contact]], axis=1)
        assert set(map(tuple, np.sort(pairs, axis=1))) == {(0, 1)}
        # merge policy reproduces the union-SDF crossing set
        sdf = _union_sdf(parts, X, Y)
        ibm_m, info_m = construct_ibm_particles(parts, x_c, contact="merge")
        assert not info_m.contact.any()
        np.testing.assert_array_equal(ibm_m.crossing_key,
                                      construct_ibm(sdf, x_c).crossing_key)
        with pytest.raises(RuntimeError, match="contact"):
            construct_ibm_particles(parts, x_c, contact="error")


def test_no_flux_decouples_particles():
    """After folding, no matrix entry couples the two touching particles."""
    N = 64
    x_f, x_c, X, Y = _grid2(N)
    parts = _touching_spheres(gap=0.0)
    ibm, info = construct_ibm_particles(parts, x_c)

    # Laplacian-like operator over the whole grid
    L = csc_array((N * N, N * N))
    for a in range(2):
        gm, _ = construct_grad((N, N), x_f, x_c[a], axis=a, format="csc")
        dv = construct_div((N, N), x_f, axis=a, format="csc")
        L = L + dv @ gm
    A, _ = apply_ibm(L.tocsr(), ibm, values_outside=0.0, values_inside=0.0)

    owner = info.owner.ravel()
    rows1 = np.flatnonzero(owner == 0)
    cols2 = np.flatnonzero(owner == 1)
    block = A[np.ix_(rows1, cols2)]
    assert np.abs(block.toarray()).max() == 0.0
    # sanity: without the contact crossings (merge) the block is nonzero
    ibm_m, _ = construct_ibm_particles(parts, x_c, contact="merge")
    A_m, _ = apply_ibm(L.tocsr(), ibm_m, values_outside=0.0, values_inside=0.0)
    assert np.abs(A_m[np.ix_(rows1, cols2)].toarray()).max() > 0.0


def test_segmentation_labels_touching():
    """Per-particle labels remain distinct at exact contact."""
    N = 48
    x_f, x_c, X, Y = _grid2(N)
    parts = _touching_spheres(gap=0.0)
    ibm, info = construct_ibm_particles(parts, x_c)
    seg = info.segmentation
    assert seg.n_segments == 2
    assert seg.sizes.sum() == int((info.owner >= 0).sum())
    labels_at = seg.labels.ravel()[ibm.row_in]
    np.testing.assert_array_equal(labels_at, info.crossing_particle + 1)


def test_contact_conditions_blending():
    N = 64
    x_f, x_c, X, Y = _grid2(N)
    parts = _touching_spheres(gap=0.0)
    ibm, info = construct_ibm_particles(parts, x_c)
    base = ({"a": (1.0, 5.0), "b": (0.0, 0.0), "d": 0.0},
            {"a": (0.0, 0.0), "b": (1.0, -1.0), "d": 0.0})
    ic = contact_conditions(base, ibm, info)
    m = info.contact
    # regular crossings keep the base coefficients
    np.testing.assert_allclose(ic[0]["a"][0][~m], 1.0)
    np.testing.assert_allclose(ic[1]["b"][0][~m], 1.0)
    # contact crossings get the default two-sided Neumann
    np.testing.assert_allclose(ic[0]["a"][0][m], 1.0)
    np.testing.assert_allclose(ic[0]["a"][1][m], 0.0)
    np.testing.assert_allclose(ic[1]["a"][1][m], 1.0)
    np.testing.assert_allclose(ic[1]["b"][0][m], 0.0)
    # no contacts -> base returned untouched
    parts_far = _touching_spheres(gap=0.05)
    ibm_f, info_f = construct_ibm_particles(parts_far, x_c)
    assert contact_conditions(base, ibm_f, info_f) is base


# ---------------------------------------------------------------------------
# Convergence regression: touching disks recover >1st order with particle
# normals (the union-SDF normals stall at ~1st order; see the study notebook)
# ---------------------------------------------------------------------------

def _solve_robin_touching(N, use_particle_normals):
    KX, KY = 1.3 * np.pi, 0.8 * np.pi

    def c_star(x, y):
        return np.cos(KX * x) * np.cos(KY * y) + 0.4 * x + 0.25 * y

    def grad_c_star(x, y):
        return (-KX * np.sin(KX * x) * np.cos(KY * y) + 0.4,
                -KY * np.cos(KX * x) * np.sin(KY * y) + 0.25)

    def lap_c_star(x, y):
        return -(KX**2 + KY**2) * np.cos(KX * x) * np.cos(KY * y)

    x_f, x_c, X, Y = _grid2(N)
    parts = _touching_spheres(gap=0.0)
    ibm, info = construct_ibm_particles(parts, x_c)
    sdf = info.pseudo_sdf

    shape = (N, N)
    diff = csc_array((N * N, N * N))
    g_bc = np.zeros((N * N, 1))
    for i in range(2):
        if i == 0:
            bc = ({'a': 0, 'b': 1, 'd': c_star(0.0, x_c[1])[None, :]},
                  {'a': 0, 'b': 1, 'd': c_star(1.0, x_c[1])[None, :]})
        else:
            bc = ({'a': 0, 'b': 1, 'd': c_star(x_c[0], 0.0)[:, None]},
                  {'a': 0, 'b': 1, 'd': c_star(x_c[0], 1.0)[:, None]})
        gm, gb = construct_grad(shape, x_f, x_c[i], bc=bc, axis=i, format='csc')
        dv = construct_div(shape, x_f, axis=i, format='csc')
        diff = diff + dv @ (-gm)
        g_bc = g_bc + dv @ (-gb)
    g_bc = np.asarray(g_bc).reshape(-1)
    f = (-lap_c_star(X, Y)).reshape(-1)

    fluid = (sdf >= 0).reshape(-1)
    diff = diff.tolil()
    for r in np.flatnonzero(~fluid):
        diff.rows[r] = [r]
        diff.data[r] = [1.0]
    diff = diff.tocsc()
    src = np.where(fluid, f - g_bc, 0.0)

    nrm = info.normals if use_particle_normals else None
    recon = construct_ibm_normal_derivative(ibm, sdf, x_c, normals=nrm)

    # exact Robin data from the exact normal
    ne = info.normals
    cx, cy = grad_c_star(ibm.coords[:, 0], ibm.coords[:, 1])
    d_rob = -(cx * ne[:, 0] + cy * ne[:, 1]) + c_star(ibm.coords[:, 0],
                                                      ibm.coords[:, 1])

    A_ibm, G_out, G_in = apply_ibm(diff, ibm, return_bc="matrix")
    H, h_vec = construct_ibm_boundary_values(
        ibm, recon, {'a': 1.0, 'b': 1.0, 'd': d_rob}, side='out')
    A = (A_ibm + G_out @ H).tocsr()
    g_tot = apply_ibm_vector(src, ibm) - np.asarray(G_out @ h_vec).ravel()
    u = spsolve(A.tocsc(), g_tot)

    e = (u - c_star(X, Y).reshape(-1))[fluid]
    return float(np.sqrt(np.sum(e**2)) / N)


def test_touching_convergence_regression():
    errs = [_solve_robin_touching(N, True) for N in (40, 80, 160)]
    orders = np.log(np.array(errs[:-1]) / np.array(errs[1:])) / np.log(2)
    assert np.all(orders > 1.4), (errs, orders)
    # and the particle normals beat the union-SDF normals at the finest grid
    err_sdf = _solve_robin_touching(160, False)
    assert errs[-1] < 0.25 * err_sdf, (errs[-1], err_sdf)
