"""Tests for the one-sided IBM interface reconstruction (:mod:`pymrm.ibm_recon`)."""

import numpy as np
import pytest

from pymrm.ibm import construct_ibm
from pymrm.ibm_recon import (
    construct_ibm_normal_derivative,
    construct_ibm_normal_derivative_ops,
    interface_normals,
    gfd_normal_derivative_weights,
    _gfd_weights_batched,
    _monomial_powers,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_axis(n, lo=0.0, hi=1.0):
    x_f = np.linspace(lo, hi, n + 1)
    return 0.5 * (x_f[1:] + x_f[:-1])


def _circle_case(n, radius=0.5, lo=-1.0, hi=1.0):
    """Solid disk of *radius* centred at the origin on an n x n grid."""
    xc = _uniform_axis(n, lo, hi)
    X, Y = np.meshgrid(xc, xc, indexing="ij")
    sdf = np.sqrt(X**2 + Y**2) - radius
    ibm = construct_ibm(sdf, [xc, xc])
    return xc, sdf, ibm


def _monomial_field(powers_row):
    """Return a monomial callable and its gradient-at-a-point for 2-D."""
    px, py = int(powers_row[0]), int(powers_row[1])

    def f(x):
        return x[..., 0] ** px * x[..., 1] ** py

    def grad_at(pt):
        gx = px * pt[0] ** (px - 1) * pt[1] ** py if px else 0.0
        gy = py * pt[0] ** px * pt[1] ** (py - 1) if py else 0.0
        return np.array([gx, gy])

    return f, grad_at


# ---------------------------------------------------------------------------
# M0: GFD weight kernel
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("degree", [1, 2])
def test_gfd_polynomial_exactness(degree):
    """Weights are exact for every monomial up to the requested degree."""
    rng = np.random.default_rng(7)
    x_gamma = np.array([0.3, -0.2])
    x_cells = x_gamma + rng.uniform(-1.0, 1.0, (14, 2)) * np.array([0.1, 2.0])
    u = np.array([0.6, -0.8])
    hvec = np.array([0.1, 2.0])          # strongly anisotropic

    alpha, gamma, info = gfd_normal_derivative_weights(
        x_gamma, x_cells, u, degree=degree, hvec=hvec)
    assert info["moment_residual"] < 1e-10

    for row in _monomial_powers(2, degree):
        f, grad_at = _monomial_field(row)
        approx = alpha * f(x_gamma) + gamma @ f(x_cells)
        exact = grad_at(x_gamma) @ u
        assert approx == pytest.approx(exact, abs=1e-10)


def test_gfd_batch_matches_single():
    """Padded batched solves equal independent single-point solves."""
    rng = np.random.default_rng(11)
    sizes = (5, 9, 13)
    dim, degree, power = 2, 2, 4
    x_gamma = np.zeros(dim)
    u = np.array([1.0, 2.0]) / np.sqrt(5.0)
    hvec = np.array([0.7, 1.3])

    cells = [rng.uniform(-1.0, 1.0, (nsz, dim)) for nsz in sizes]
    singles = [gfd_normal_derivative_weights(
        x_gamma, c, u, degree=degree, hvec=hvec, weight_power=power)
        for c in cells]

    n1 = 1 + max(sizes)
    Y = np.zeros((len(sizes), n1, dim))
    qinv = np.zeros((len(sizes), n1))
    for i, c in enumerate(cells):
        Yk = c / hvec
        Y[i, 1:1 + c.shape[0]] = Yk
        r = np.linalg.norm(Yk, axis=1)
        qinv[i, 1:1 + c.shape[0]] = 1.0 / (1.0 + r ** power)
        qinv[i, 0] = 1.0
    u_scaled = np.broadcast_to(u / hvec, (len(sizes), dim))

    a, cond, mres = _gfd_weights_batched(Y, qinv, u_scaled, degree)
    for i, (alpha, gamma, info) in enumerate(singles):
        assert a[i, 0] == pytest.approx(alpha, rel=1e-12)
        np.testing.assert_allclose(a[i, 1:1 + sizes[i]], gamma, rtol=1e-12)
        assert np.all(a[i, 1 + sizes[i]:] == 0.0)
        assert cond[i] == pytest.approx(info["cond"], rel=1e-8)


def test_two_point_formula_linear_exact():
    """degree=0 forces the two-point normal formula; exact for c = a + b*x."""
    x_c = _uniform_axis(20)
    sdf = 0.63 - x_c                       # fluid left, solid right
    ibm = construct_ibm(sdf, x_c)
    recon = construct_ibm_normal_derivative(ibm, sdf, x_c, degree=0)

    assert np.all(recon.degree_out == 0)
    assert np.all(recon.degree_in == 0)
    assert np.all(recon.alpha_out > 0)
    assert np.all(recon.alpha_in > 0)

    c = 2.0 + 3.0 * x_c
    w = 2.0 + 3.0 * ibm.coords[:, 0]
    q_out = recon.alpha_out * w + recon.D_out @ c
    q_in = recon.alpha_in * w + recon.D_in @ c
    # Fluid outward is +x, solid outward is -x.
    np.testing.assert_allclose(q_out, 3.0, rtol=1e-12)
    np.testing.assert_allclose(q_in, -3.0, rtol=1e-12)


def test_degree_demotion_few_points():
    """Too few same-side cells lowers the degree instead of failing."""
    x_c = _uniform_axis(6)                 # cells at 1/12, 3/12, 5/12, ...
    sdf = 0.3 - x_c                        # 2 fluid cells, 4 solid cells
    ibm = construct_ibm(sdf, x_c)
    recon = construct_ibm_normal_derivative(ibm, sdf, x_c, degree=2)
    # p=2 in 1-D needs ceil(1.5*2)=3 points; the fluid side only has 2 cells
    # in the whole domain, so even the enlarged radius cannot rescue p=2.
    assert recon.degree_out[0] == 1
    # The solid side has 4 cells: p=2 succeeds (possibly after enlarging).
    assert recon.degree_in[0] == 2

    # Even a single same-side cell must not fail (two-point formula).
    sdf1 = 0.2 - x_c                       # 1 fluid cell
    with pytest.warns(RuntimeWarning):     # cut cell at the domain boundary
        ibm1 = construct_ibm(sdf1, x_c)
    recon1 = construct_ibm_normal_derivative(ibm1, sdf1, x_c, degree=2)
    assert recon1.degree_out[0] == 0
    assert recon1.n_stencil_out[0] == 1


# ---------------------------------------------------------------------------
# M1: reconstruction layer on immersed geometries
# ---------------------------------------------------------------------------

def test_circle_normals():
    """SDF-gradient normals of a circle match the radial direction at O(h^2)."""
    errors = []
    for n in (40, 80):
        xc, sdf, ibm = _circle_case(n)
        normals = interface_normals(ibm, sdf, [xc, xc])
        radial = ibm.coords / np.linalg.norm(ibm.coords, axis=1, keepdims=True)
        # Solid inside, fluid outside: n points outward (radially).
        errors.append(np.abs(normals - radial).max())
    assert errors[0] < 5e-3
    assert errors[1] < 0.35 * errors[0]


def test_stencil_purity():
    """D_out only touches fluid cells; D_in only touches solid cells."""
    xc, sdf, ibm = _circle_case(40)
    recon = construct_ibm_normal_derivative(ibm, sdf, [xc, xc])
    fluid = sdf.ravel() >= 0
    out_cols = recon.D_out.tocoo().col
    in_cols = recon.D_in.tocoo().col
    assert np.all(fluid[out_cols])
    assert np.all(~fluid[in_cols])


def test_flood_fill_excludes_across_thin_solid():
    """Fluid cells on the far side of a thin plate are never used."""
    n = 40
    xc = _uniform_axis(n)                  # h = 0.025
    h = xc[1] - xc[0]
    X, Y = np.meshgrid(xc, xc, indexing="ij")
    # Thin vertical plate, one cell thick, from the bottom up to y = 0.6:
    # the fluid wraps around the top, so the global fluid region is one
    # connected component and only the local flood fill separates the sides.
    x_plate = xc[n // 2]
    half_w = 0.6 * h
    dx = np.abs(X - x_plate) - half_w
    dy = Y - 0.6
    sdf = np.where(dy <= 0, dx, np.maximum(dx, dy))
    ibm = construct_ibm(sdf, [xc, xc])
    recon = construct_ibm_normal_derivative(ibm, sdf, [xc, xc])

    # Crossings along the plate axis, well below the top edge.
    plate_side = ibm.axis == 0
    low = ibm.coords[:, 1] < 0.4
    sel = plate_side & low
    assert np.any(sel)

    coo = recon.D_out.tocoo()
    x_cells = xc[np.unravel_index(coo.col, (n, n))[0]]
    for k in np.flatnonzero(sel):
        mask = coo.row == k
        assert np.any(mask)
        # All stencil cells on the same side of the plate as the wall point.
        side = np.sign(ibm.coords[k, 0] - x_plate)
        assert np.all(np.sign(x_cells[mask] - x_plate) == side), (
            f"crossing {k}: stencil leaks across the thin plate")

    # Control: without any visibility filter the leak does occur.
    recon_none = construct_ibm_normal_derivative(
        ibm, sdf, [xc, xc], connectivity="none")
    coo_n = recon_none.D_out.tocoo()
    x_cells_n = xc[np.unravel_index(coo_n.col, (n, n))[0]]
    leaked = False
    for k in np.flatnonzero(sel):
        mask = coo_n.row == k
        side = np.sign(ibm.coords[k, 0] - x_plate)
        if np.any(np.sign(x_cells_n[mask] - x_plate) != side):
            leaked = True
            break
    assert leaked, "control without filter should show cross-plate stencils"


@pytest.mark.parametrize("degree,min_order", [(2, 1.7), (1, 0.8)])
def test_normal_derivative_convergence(degree, min_order):
    """q_side converges to the analytic directional derivative at order ~p."""
    def field(x):
        return np.exp(0.8 * x[..., 0]) * np.sin(1.3 * x[..., 1])

    def grad(x):
        gx = 0.8 * np.exp(0.8 * x[..., 0]) * np.sin(1.3 * x[..., 1])
        gy = 1.3 * np.exp(0.8 * x[..., 0]) * np.cos(1.3 * x[..., 1])
        return np.stack([gx, gy], axis=-1)

    errors = []
    for n in (20, 40, 80):
        xc, sdf, ibm = _circle_case(n)
        recon = construct_ibm_normal_derivative(ibm, sdf, [xc, xc],
                                             degree=degree)
        X, Y = np.meshgrid(xc, xc, indexing="ij")
        pts = np.stack([X, Y], axis=-1)
        c = field(pts).ravel()
        w = field(ibm.coords)
        q_out = recon.alpha_out * w + recon.D_out @ c
        u_out = -recon.normals                      # fluid outward
        exact = np.einsum("kd,kd->k", grad(ibm.coords), u_out)
        errors.append(np.abs(q_out - exact).max())
    orders = np.log(np.array(errors[:-1]) / np.array(errors[1:])) / np.log(2)
    assert np.all(orders > min_order), (errors, orders)


def test_length_scale_cap():
    """A small length scale caps the radius and demotes the degree."""
    xc, sdf, ibm = _circle_case(40)
    h = xc[1] - xc[0]
    L = 2.0 * h
    recon = construct_ibm_normal_derivative(ibm, sdf, [xc, xc],
                                         degree=2, length_scale=L)
    cap = 0.5 * L
    assert np.all(recon.radius_out <= cap * (1 + 1e-12))
    assert np.all(recon.radius_in <= cap * (1 + 1e-12))
    # Under the cap there is no room for quadratic stencils: degrees demote
    # rather than the stencil reaching farther.
    assert np.all(recon.degree_out < 2)
    assert np.all(recon.degree_in < 2)
    # Reference: without the cap, quadratic reconstruction succeeds.
    recon_free = construct_ibm_normal_derivative(ibm, sdf, [xc, xc], degree=2)
    assert np.all(recon_free.degree_out == 2)


def test_normal_derivative_ops_ns_expansion():
    """Full-field operators reproduce per-component spatial results."""
    n, ns = 30, 3
    xc = _uniform_axis(n)
    sdf = 0.55 - xc
    ibm = construct_ibm(sdf, xc, axes=(0,), shape=(n, ns))
    recon = construct_ibm_normal_derivative(ibm, sdf, xc)
    alpha_out_full, N_out, alpha_in_full, N_in = \
        construct_ibm_normal_derivative_ops(ibm, recon)

    assert N_out.shape == (ibm.n_crossings * ns, n * ns)
    np.testing.assert_array_equal(alpha_out_full,
                                  np.repeat(recon.alpha_out, ns))

    rng = np.random.default_rng(4)
    c = rng.normal(size=(n, ns))
    q_full = (N_out @ c.ravel()).reshape(ibm.n_crossings, ns)
    for j in range(ns):
        np.testing.assert_allclose(q_full[:, j], recon.D_out @ c[:, j],
                                   rtol=1e-13)
