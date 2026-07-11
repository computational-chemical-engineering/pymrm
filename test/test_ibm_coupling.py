"""Tests for general immersed interface conditions (:mod:`pymrm.ibm_coupling`)."""

import numpy as np
import pytest
from scipy.sparse import diags_array, kron, csr_array
from scipy.sparse.linalg import spsolve

from pymrm.operators import construct_grad, construct_div
from pymrm.coupling import construct_interface_matrices
from pymrm.ibm import construct_ibm, apply_ibm, apply_ibm_vector, _expand_full
from pymrm.ibm_recon import construct_ibm_normal_derivative
from pymrm.ibm_coupling import (
    construct_ibm_interface_values,
    apply_ibm_interface,
    construct_ibm_boundary_values,
)
from scipy.sparse import coo_array


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_axis(n, lo=0.0, hi=1.0):
    x_f = np.linspace(lo, hi, n + 1)
    return x_f, 0.5 * (x_f[1:] + x_f[:-1])


def _diffusion_1d(n, x_f, x_c, D, bc):
    """1-D operator ``L = div(-D grad)`` and its *added* boundary source."""
    grad, grad_bc = construct_grad(n, x_f, x_c, bc=bc)
    div = construct_div(n, x_f)
    L = (div @ (-D * grad)).tocsr()
    g = (div @ (-D * grad_bc)).toarray().ravel()
    return L, g


def _mix_rows(fluid, L_f, g_f, L_s, g_s):
    """Row-wise combination: fluid rows from (L_f, g_f), solid from (L_s, g_s)."""
    sel_f = diags_array(fluid.astype(float))
    sel_s = diags_array((~fluid).astype(float))
    return (sel_f @ L_f + sel_s @ L_s).tocsr(), np.where(fluid, g_f, g_s)


def _conjugate_ic(D_out, D_in, K=1.0):
    return ({"a": (D_out, D_in), "b": (0.0, 0.0), "d": 0.0},
            {"a": (0.0, 0.0), "b": (1.0, -K), "d": 0.0})


def _solve_conjugate_1d(n, x_w, D_f, D_s, K=1.0, S_solid=0.0, c0=0.0, c1=1.0,
                        degree=2, rescale=True):
    """Immersed 1-D conjugate diffusion: fluid left of *x_w*, solid right."""
    x_f, x_c = _uniform_axis(n)
    bc = ({"a": 0, "b": 1, "d": c0}, {"a": 0, "b": 1, "d": c1})
    L_f, g_f = _diffusion_1d(n, x_f, x_c, D_f, bc)
    L_s, g_s = _diffusion_1d(n, x_f, x_c, D_s, bc)
    sdf = x_w - x_c
    fluid = sdf >= 0
    L, g = _mix_rows(fluid, L_f, g_f, L_s, g_s)

    ibm = construct_ibm(sdf, x_c, rescale=rescale)
    recon = construct_ibm_normal_derivative(ibm, sdf, x_c, degree=degree)
    A, g_ic = apply_ibm_interface(L, ibm, recon, _conjugate_ic(D_f, D_s, K))

    src = np.where(fluid, 0.0, S_solid)
    rhs = apply_ibm_vector(src - g, ibm) - g_ic
    return x_c, spsolve(A.tocsc(), rhs), sdf


# ---------------------------------------------------------------------------
# (a) Machine-precision equivalence with construct_interface_matrices
# ---------------------------------------------------------------------------

def test_matches_construct_interface_matrices():
    """Immersed elimination equals the grid-aligned reference exactly.

    With the wall on a grid face and the GFD stencil restricted to the two
    same-side cells + interface node, the determined quadratic is unique, so
    the interface values must equal ``construct_interface_matrices`` for the
    same ic dictionaries -- for an arbitrary field and inhomogeneous ic.
    """
    n = 12
    x_f, x_c = _uniform_axis(n)
    n0 = n // 2                              # wall exactly on the face x=0.5

    ic = ({"a": (1.3, 2.7), "b": (0.2, -0.1), "d": 0.5},
          {"a": (0.4, -0.9), "b": (1.0, -1.7), "d": -0.25})

    im0, ibc0, im1, ibc1 = construct_interface_matrices(
        ((n0,), (n - n0,)),
        (np.linspace(0.0, 0.5, n0 + 1), np.linspace(0.5, 1.0, n - n0 + 1)),
        ic=ic,
    )

    sdf = 0.5 - x_c                          # fluid = subdomain 0 (left)
    ibm = construct_ibm(sdf, x_c)
    recon = construct_ibm_normal_derivative(ibm, sdf, x_c,
                                         min_points_factor=1.0)
    assert np.all(recon.degree_out == 2) and np.all(recon.degree_in == 2)
    assert recon.n_stencil_out[0] == 2 and recon.n_stencil_in[0] == 2

    H_out, h_out, H_in, h_in = construct_ibm_interface_values(ibm, recon, ic)

    rng = np.random.default_rng(5)
    c = rng.normal(size=n)
    ref_out = np.asarray(im0 @ c).ravel() + ibc0.toarray().ravel()
    ref_in = np.asarray(im1 @ c).ravel() + ibc1.toarray().ravel()
    np.testing.assert_allclose(H_out @ c + h_out, ref_out, rtol=1e-9)
    np.testing.assert_allclose(H_in @ c + h_in, ref_in, rtol=1e-9)


# ---------------------------------------------------------------------------
# (b) Manufactured 1-D solutions
# ---------------------------------------------------------------------------

def test_conjugate_1d_piecewise_linear_exact():
    """No source: the piecewise-linear solution is reproduced exactly."""
    D_f, D_s, x_w = 1.0, 5.0, 0.6
    x_c, c, _ = _solve_conjugate_1d(40, x_w, D_f, D_s)
    J = 1.0 / (x_w / D_f + (1.0 - x_w) / D_s)
    exact = np.where(x_c < x_w, J * x_c / D_f,
                     J * x_w / D_f + J * (x_c - x_w) / D_s)
    np.testing.assert_allclose(c, exact, atol=1e-12)


@pytest.mark.parametrize("degree,tol,should_fail_tol", [(2, 1e-9, None),
                                                        (0, None, 1e-6)])
def test_conjugate_1d_quadratic_manufactured(degree, tol, should_fail_tol):
    """Solid source: quadratic solution is exact for p=2, not for two-point."""
    n, x_w, D_f, D_s, S = 40, 0.6123, 1.0, 3.0, 4.0
    m = 3.0                                     # fluid solution c = 2 + 3 x
    c_at = lambda xw: 2.0 + m * xw

    def exact(x):
        solid = x >= x_w
        cs = (c_at(x_w) + (D_f * m / D_s) * (x - x_w)
              - (S / (2 * D_s)) * (x - x_w) ** 2)
        return np.where(solid, cs, 2.0 + m * x)

    x_f, x_c = _uniform_axis(n)
    bc = ({"a": 0, "b": 1, "d": exact(np.array([0.0]))[0]},
          {"a": 0, "b": 1, "d": exact(np.array([1.0]))[0]})
    L_f, g_f = _diffusion_1d(n, x_f, x_c, D_f, bc)
    L_s, g_s = _diffusion_1d(n, x_f, x_c, D_s, bc)
    sdf = x_w - x_c
    fluid = sdf >= 0
    L, g = _mix_rows(fluid, L_f, g_f, L_s, g_s)
    ibm = construct_ibm(sdf, x_c)
    recon = construct_ibm_normal_derivative(ibm, sdf, x_c, degree=degree)
    A, g_ic = apply_ibm_interface(L, ibm, recon, _conjugate_ic(D_f, D_s))
    src = np.where(fluid, 0.0, S)
    c = spsolve(A.tocsc(), apply_ibm_vector(src - g, ibm) - g_ic)

    err = np.abs(c - exact(x_c)).max()
    if tol is not None:
        assert err < tol
    else:
        assert err > should_fail_tol           # two-point tier is not exact


# ---------------------------------------------------------------------------
# (c) 2-D circle conjugate diffusion vs analytic two-region solution
# ---------------------------------------------------------------------------

def _solve_circle_conjugate(n, degree):
    """Solid disk with a uniform source in a fluid square, full conjugate."""
    R, D_f, D_s, S, B = 0.35, 1.0, 2.5, 4.0, 1.0
    C = -S * R**2 / (2.0 * D_f)
    A_const = B + C * np.log(R) + S * R**2 / (4.0 * D_s)

    def c_fluid(r):
        return B + C * np.log(np.maximum(r, 0.5 * R))

    def c_solid(r):
        return A_const - S * r**2 / (4.0 * D_s)

    x_f, x_c = _uniform_axis(n, -1.0, 1.0)
    X, Y = np.meshgrid(x_c, x_c, indexing="ij")
    r_c = np.sqrt(X**2 + Y**2)
    sdf = r_c - R
    fluid = (sdf >= 0).ravel()
    shape = (n, n)

    def assemble(D):
        L = None
        g = np.zeros(n * n)
        for axis in range(2):
            if axis == 0:
                d_lo = c_fluid(np.sqrt(x_f[0]**2 + x_c**2))[np.newaxis, :]
                d_hi = c_fluid(np.sqrt(x_f[-1]**2 + x_c**2))[np.newaxis, :]
            else:
                d_lo = c_fluid(np.sqrt(x_c**2 + x_f[0]**2))[:, np.newaxis]
                d_hi = c_fluid(np.sqrt(x_c**2 + x_f[-1]**2))[:, np.newaxis]
            bc = ({"a": 0, "b": 1, "d": d_lo}, {"a": 0, "b": 1, "d": d_hi})
            grad, grad_bc = construct_grad(shape, x_f, x_c, bc=bc, axis=axis)
            div = construct_div(shape, x_f, axis=axis)
            term = (div @ (-D * grad)).tocsr()
            L = term if L is None else L + term
            g += (div @ (-D * grad_bc)).toarray().ravel()
        return L, g

    L_f, g_f = assemble(D_f)
    L_s, g_s = assemble(D_s)
    L, g = _mix_rows(fluid, L_f, g_f, L_s, g_s)

    ibm = construct_ibm(sdf, [x_c, x_c])
    recon = construct_ibm_normal_derivative(ibm, sdf, [x_c, x_c], degree=degree)
    A_mat, g_ic = apply_ibm_interface(L, ibm, recon,
                                      _conjugate_ic(D_f, D_s))
    src = np.where(fluid, 0.0, S)
    c = spsolve(A_mat.tocsc(), apply_ibm_vector(src - g, ibm) - g_ic)

    exact = np.where(fluid, c_fluid(r_c).ravel(), c_solid(r_c).ravel())
    return np.abs(c - exact).max()


@pytest.mark.parametrize("degree,min_order", [(2, 1.6), (0, 0.6)])
def test_circle_conjugate_convergence(degree, min_order):
    errors = [_solve_circle_conjugate(n, degree) for n in (40, 80, 160)]
    orders = np.log(np.array(errors[:-1]) / np.array(errors[1:])) / np.log(2)
    assert np.all(orders > min_order), (errors, orders)


# ---------------------------------------------------------------------------
# (d) Row-scale consistency
# ---------------------------------------------------------------------------

def test_rescale_invariance():
    _, c_scaled, _ = _solve_conjugate_1d(40, 0.6123, 1.0, 5.0, S_solid=3.0,
                                         rescale=True)
    _, c_plain, _ = _solve_conjugate_1d(40, 0.6123, 1.0, 5.0, S_solid=3.0,
                                        rescale=False)
    np.testing.assert_allclose(c_scaled, c_plain, atol=1e-9)


# ---------------------------------------------------------------------------
# (e) Multi-component fields (ns > 1) with per-component coefficients
# ---------------------------------------------------------------------------

def test_multicomponent_partition():
    """(n, 2) field with per-component D and K equals two scalar solves."""
    n, x_w = 40, 0.6
    D_f = np.array([1.0, 2.0])
    D_s = np.array([5.0, 0.5])
    K = np.array([1.0, 2.5])

    x_f, x_c = _uniform_axis(n)
    sdf = x_w - x_c
    fluid = sdf >= 0

    # Reference: independent scalar solves.
    ref = [
        _solve_conjugate_1d(n, x_w, D_f[j], D_s[j], K=K[j])[1]
        for j in range(2)
    ]

    # Full (n, 2) system: component j on the ns axis via kron with E_jj.
    bc = ({"a": 0, "b": 1, "d": 0.0}, {"a": 0, "b": 1, "d": 1.0})
    L_full = None
    g_full = np.zeros(2 * n)
    for j in range(2):
        E = np.zeros((2, 2))
        E[j, j] = 1.0
        L_f, g_f = _diffusion_1d(n, x_f, x_c, D_f[j], bc)
        L_s, g_s = _diffusion_1d(n, x_f, x_c, D_s[j], bc)
        L_j, g_j = _mix_rows(fluid, L_f, g_f, L_s, g_s)
        term = csr_array(kron(L_j, E))
        L_full = term if L_full is None else L_full + term
        g_j2 = np.zeros((n, 2))
        g_j2[:, j] = g_j
        g_full += g_j2.ravel()

    ibm = construct_ibm(sdf, x_c, axes=(0,), shape=(n, 2))
    recon = construct_ibm_normal_derivative(ibm, sdf, x_c)
    ic = ({"a": (D_f[np.newaxis, :], D_s[np.newaxis, :]),
           "b": (0.0, 0.0), "d": 0.0},
          {"a": (0.0, 0.0),
           "b": (np.ones((1, 2)), -K[np.newaxis, :]), "d": 0.0})
    A, g_ic = apply_ibm_interface(L_full, ibm, recon, ic)
    rhs = apply_ibm_vector(-g_full, ibm) - g_ic
    c = spsolve(A.tocsc(), rhs).reshape(n, 2)

    np.testing.assert_allclose(c[:, 0], ref[0], atol=1e-10)
    np.testing.assert_allclose(c[:, 1], ref[1], atol=1e-10)


# ---------------------------------------------------------------------------
# (f) Degenerate ic reduces to the Dirichlet IBM; single-side Robin
# ---------------------------------------------------------------------------

def test_dirichlet_degenerate_ic():
    """ic pinning both interface values reproduces plain apply_ibm."""
    n = 24
    x_f, x_c = _uniform_axis(n)
    sdf = 0.57 - x_c
    bc = ({"a": 0, "b": 1, "d": 0.0}, {"a": 0, "b": 1, "d": 1.0})
    L, _ = _diffusion_1d(n, x_f, x_c, 1.0, bc)

    ibm = construct_ibm(sdf, x_c)
    recon = construct_ibm_normal_derivative(ibm, sdf, x_c)
    rng = np.random.default_rng(8)
    v_out = rng.normal(size=ibm.n_crossings)
    v_in = rng.normal(size=ibm.n_crossings)

    A_ref, g_ref = apply_ibm(L, ibm, values_outside=v_out, values_inside=v_in)
    ic = ({"a": (0.0, 0.0), "b": (1.0, 0.0), "d": v_out},
          {"a": (0.0, 0.0), "b": (0.0, 1.0), "d": v_in})
    A, g = apply_ibm_interface(L, ibm, recon, ic)

    assert np.abs((A - A_ref).toarray()).max() < 1e-12
    np.testing.assert_allclose(g, g_ref, atol=1e-12)


def test_single_side_robin():
    """Immersed Robin wall (fluid only): exact for the linear solution."""
    n, x_w, D, k_mt = 40, 0.6, 1.0, 2.0
    x_f, x_c = _uniform_axis(n)
    sdf = x_w - x_c
    fluid = sdf >= 0
    bc = ({"a": 0, "b": 1, "d": 1.0}, {"a": 0, "b": 1, "d": 0.0})
    L_f, g_f = _diffusion_1d(n, x_f, x_c, D, bc)
    eye = diags_array(np.ones(n)).tocsr()
    L, g = _mix_rows(fluid, L_f, g_f, eye, np.zeros(n))

    ibm = construct_ibm(sdf, x_c)
    recon = construct_ibm_normal_derivative(ibm, sdf, x_c)
    A_ibm, G_out, G_in = apply_ibm(L, ibm, return_bc="matrix")
    # D * q_out + k_mt * c_gamma = 0  (mass transfer into an unmodeled solid)
    H, h = construct_ibm_boundary_values(ibm, recon,
                                         {"a": D, "b": k_mt, "d": 0.0})
    A = (A_ibm + G_out @ H).tocsr()
    g_tot = apply_ibm_vector(g, ibm) + np.asarray(G_out @ h).ravel()
    c = spsolve(A.tocsc(), -g_tot)

    # Analytic: c = 1 + m x with D*m + k_mt*(1 + m*x_w) = 0.
    m = -k_mt / (D + k_mt * x_w)
    exact = 1.0 + m * x_c
    np.testing.assert_allclose(c[fluid], exact[fluid], atol=1e-10)


# ---------------------------------------------------------------------------
# (g) Sandwich: one-cell-thick solid between two fluid regions
# ---------------------------------------------------------------------------

def test_sandwich_thin_solid_conjugate():
    """A 1-cell solid wall with conjugate conditions stays exact for the
    piecewise-linear solution (sibling wall-value routing through G @ H)."""
    n, D_f, D_s = 30, 1.0, 0.2
    x_f, x_c = _uniform_axis(n)
    k = 3 * n // 5
    xm = x_c[k]
    w = 1.2 * (x_c[1] - x_c[0])
    sdf = np.abs(x_c - xm) - 0.5 * w
    assert np.sum(sdf < 0) == 1              # exactly one solid cell

    fluid = sdf >= 0
    bc = ({"a": 0, "b": 1, "d": 0.0}, {"a": 0, "b": 1, "d": 1.0})
    L_f, g_f = _diffusion_1d(n, x_f, x_c, D_f, bc)
    L_s, g_s = _diffusion_1d(n, x_f, x_c, D_s, bc)
    L, g = _mix_rows(fluid, L_f, g_f, L_s, g_s)

    ibm = construct_ibm(sdf, x_c)
    assert np.any(ibm.sib_in >= 0)           # solid side is a sandwich
    recon = construct_ibm_normal_derivative(ibm, sdf, x_c)
    assert np.all(recon.degree_in <= 1)      # single-cell stencils flagged

    A, g_ic = apply_ibm_interface(L, ibm, recon, _conjugate_ic(D_f, D_s))
    c = spsolve(A.tocsc(), apply_ibm_vector(-g, ibm) - g_ic)

    a, b = xm - 0.5 * w, xm + 0.5 * w
    J = 1.0 / (a / D_f + w / D_s + (1.0 - b) / D_f)
    exact = np.where(
        x_c < a, J * x_c / D_f,
        np.where(x_c > b,
                 1.0 - J * (1.0 - x_c) / D_f,
                 J * a / D_f + J * (x_c - a) / D_s))
    assert np.all(np.isfinite(c))
    np.testing.assert_allclose(c, exact, atol=1e-10)


# ---------------------------------------------------------------------------
# (h) Multi-dimensional ic-coefficient broadcasting (interleaved ns axes)
# ---------------------------------------------------------------------------

def _circle_ibm_recon(n=16, npn=2, ncn=3, radius=0.28):
    """IBM + reconstruction on an interleaved field (nx, np, ny, nc)."""
    x_f = np.linspace(0.0, 1.0, n + 1)
    x_c = 0.5 * (x_f[1:] + x_f[:-1])
    xx, yy = np.meshgrid(x_c, x_c, indexing="ij")
    sdf = np.hypot(xx - 0.5, yy - 0.5) - radius
    ibm = construct_ibm(sdf, [x_c, x_c], axes=(0, 2), shape=(n, npn, n, ncn))
    recon = construct_ibm_normal_derivative(ibm, sdf, [x_c, x_c])
    return ibm, recon


def _ghost_extraction_full(ibm, value=1.0):
    fr = np.concatenate([_expand_full(ibm, ibm.row_out).ravel(),
                         _expand_full(ibm, ibm.row_in).ravel()])
    fg = np.concatenate([_expand_full(ibm, ibm.ghost_out).ravel(),
                         _expand_full(ibm, ibm.ghost_in).ravel()])
    data = np.full(fr.size, float(value))
    return coo_array((data, (fr, fg)),
                     shape=(ibm.n_cells, ibm.n_cells)).tocsr()


def test_ic_coeff_multidim_broadcast_equals_explicit():
    """(np, nc)-shaped ic coefficients equal their explicit (npnt, np, nc) form."""
    npn, ncn = 2, 3
    ibm, recon = _circle_ibm_recon(npn=npn, ncn=ncn)
    npnt = ibm.n_crossings
    rng = np.random.default_rng(11)
    D_out = rng.random((npn, ncn)) + 0.5
    D_in = rng.random((npn, ncn)) + 0.5
    K = rng.random((npn, ncn)) + 0.5
    dval = rng.random((npn, ncn))

    ic_bcast = ({"a": (D_out, D_in), "b": (0.0, 0.0), "d": 0.0},
                {"a": (0.0, 0.0), "b": (np.ones((npn, ncn)), -K), "d": dval})

    def expl(v):
        return np.broadcast_to(np.asarray(v, dtype=float),
                               (npnt, npn, ncn)).copy()

    ic_expl = ({"a": (expl(D_out), expl(D_in)), "b": (expl(0.0), expl(0.0)),
                "d": expl(0.0)},
               {"a": (expl(0.0), expl(0.0)),
                "b": (expl(np.ones((npn, ncn))), expl(-K)), "d": expl(dval)})

    Ho1, ho1, Hi1, hi1 = construct_ibm_interface_values(ibm, recon, ic_bcast)
    Ho2, ho2, Hi2, hi2 = construct_ibm_interface_values(ibm, recon, ic_expl)
    assert np.abs((Ho1 - Ho2).toarray()).max() < 1e-12
    assert np.abs((Hi1 - Hi2).toarray()).max() < 1e-12
    np.testing.assert_allclose(ho1, ho2, atol=1e-12)
    np.testing.assert_allclose(hi1, hi2, atol=1e-12)

    # And the full assembly matches on a matrix with nonzero ghost entries.
    A0 = _ghost_extraction_full(ibm)
    A1, g1 = apply_ibm_interface(A0, ibm, recon, ic_bcast)
    A2, g2 = apply_ibm_interface(A0, ibm, recon, ic_expl)
    assert np.abs((A1 - A2).toarray()).max() < 1e-12
    np.testing.assert_allclose(g1, g2, atol=1e-12)


def test_ic_coeff_bare_1d_with_ns_axes_errors():
    """A bare (npnt,) ic coefficient is rejected when ns axes are present."""
    ibm, recon = _circle_ibm_recon(npn=2, ncn=3)
    npnt = ibm.n_crossings
    assert npnt != 3
    ic = ({"a": (np.ones(npnt), np.ones(npnt)), "b": (0.0, 0.0), "d": 0.0},
          {"a": (0.0, 0.0), "b": (1.0, -1.0), "d": 0.0})
    with pytest.raises(ValueError, match="per-crossing"):
        construct_ibm_interface_values(ibm, recon, ic)
