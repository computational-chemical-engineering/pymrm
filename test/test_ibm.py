"""Tests for the directional ghost-cell immersed boundary method (:mod:`pymrm.ibm`)."""

import numpy as np
import pytest
from scipy.sparse import csr_array, lil_array
from scipy.sparse.linalg import spsolve

from pymrm.ibm import construct_ibm, apply_ibm, apply_ibm_vector
from pymrm.grid import generate_grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_1d(n=10, x_wall=0.63):
    """Uniform 1-D grid on [0, 1] with a planar immersed wall at ``x_wall``.

    Fluid is ``x < x_wall`` (``sdf > 0``); solid is ``x > x_wall``.
    """
    x_f = np.linspace(0.0, 1.0, n + 1)
    _, x_c = generate_grid(n, x_f, generate_x_c=True)
    sdf = x_wall - x_c
    return x_f, x_c, sdf


def _ref_point_value(theta):
    """Reference equidistant point-value coefficients (center, opp, wall)."""
    d = theta * (1.0 + theta)
    c_center = 2.0 * (theta**2 - 1.0) / d
    c_opp = theta * (1.0 - theta) / d
    c_wall = 2.0 / d
    return c_center, c_opp, c_wall


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def test_classification_two_sided():
    x_f, x_c, sdf = _uniform_1d(n=10, x_wall=0.63)
    ibm = construct_ibm(sdf, x_f)

    # Fluid cells: x_c < 0.63 -> indices 0..5 ; cut fluid cell is 5, ghost 6.
    assert ibm.outside.n_points == 1
    assert ibm.outside.row[0] == 5
    assert ibm.outside.ghost[0] == 6
    assert ibm.outside.opp[0] == 4
    assert ibm.outside.direction[0] == 1

    # Inside (solid) cut cell is 6, ghost is the fluid cell 5.
    assert ibm.inside.n_points == 1
    assert ibm.inside.row[0] == 6
    assert ibm.inside.ghost[0] == 5
    assert ibm.inside.direction[0] == -1


def test_crossing_key_alignment():
    x_f, x_c, sdf = _uniform_1d(n=10, x_wall=0.63)
    ibm = construct_ibm(sdf, x_f)
    # Outside and inside points on the same face share the crossing key.
    assert sorted(ibm.outside.crossing_key.tolist()) == sorted(
        ibm.inside.crossing_key.tolist()
    )


# ---------------------------------------------------------------------------
# Coefficient correctness
# ---------------------------------------------------------------------------

def test_equidistant_reduction_matches_reference():
    # Wall at 0.63 on a uniform grid h=0.1 -> theta = 0.8 for cut cell 5.
    x_f, x_c, sdf = _uniform_1d(n=10, x_wall=0.63)
    ibm = construct_ibm(sdf, x_f, rescale=False)
    s = ibm.outside
    theta = (0.63 - x_c[5]) / (x_c[6] - x_c[5])
    c_center, c_opp, c_wall = _ref_point_value(theta)
    assert s.coef_c[0] == pytest.approx(c_center)
    assert s.coef_o[0] == pytest.approx(c_opp)
    assert s.coef_w_self[0] == pytest.approx(c_wall)


def test_reconstruction_second_order_convergence():
    # The 3-node point-value reconstruction is exact for quadratics, so the
    # ghost-value error for a smooth field decreases at ~3rd order in h. Keep
    # theta = 0.5 fixed across refinements so the error constant is stable.
    def field(x):
        return np.exp(1.3 * x)

    errors = []
    for n in (20, 40, 80, 160):
        x_f = np.linspace(0.0, 1.0, n + 1)
        _, x_c = generate_grid(n, x_f, generate_x_c=True)
        k = n // 2
        h = x_c[k + 1] - x_c[k]
        x_wall = x_c[k] + 0.5 * h  # theta = 0.5 for cut cell k
        sdf = x_wall - x_c
        ibm = construct_ibm(sdf, x_f, rescale=False)
        s = ibm.outside
        u = field(x_c)
        u_wall = field(s.coords[:, 0])

        # Extraction operator: row c reads the ghost neighbor value.
        n_cells = x_c.size
        A = lil_array((n_cells, n_cells))
        for c, g in zip(s.row, s.ghost):
            A[c, g] = 1.0
        A = csr_array(A)

        A_mod, bc = apply_ibm(A, ibm, side="outside", values=u_wall)
        recon = (A_mod @ u) + bc
        err = abs(recon[k] - field(x_c[k + 1]))
        errors.append(err)

    errors = np.array(errors)
    orders = np.log(errors[:-1] / errors[1:]) / np.log(2.0)
    assert np.all(orders > 2.5)


def test_nonuniform_quadratic_exact():
    # On a non-uniform grid the reconstruction must be exact for a quadratic.
    n = 15
    x_f = np.sort(np.concatenate(([0.0, 1.0], np.random.default_rng(0).uniform(0, 1, n - 1))))
    _, x_c = generate_grid(n, x_f, generate_x_c=True)
    x_wall = 0.6123
    sdf = x_wall - x_c
    ibm = construct_ibm(sdf, x_f, rescale=False)
    s = ibm.outside

    def quad(x):
        return 1.0 - 2.0 * x + 3.0 * x**2

    u = quad(x_c)
    u_wall = quad(s.coords[:, 0])
    recon = (
        s.coef_c * u[s.row]
        + s.coef_o * np.where(s.opp >= 0, u[np.where(s.opp >= 0, s.opp, 0)], 0.0)
        + s.coef_w_self * u_wall
    )
    expected = quad(x_c[s.ghost])
    assert np.allclose(recon, expected)


# ---------------------------------------------------------------------------
# Sandwich (thin solid)
# ---------------------------------------------------------------------------

def test_sandwich_structure_and_exactness():
    # A single solid cell surrounded by fluid -> inside side is a sandwich.
    n = 11
    x_f = np.linspace(0.0, 1.0, n + 1)
    _, x_c = generate_grid(n, x_f, generate_x_c=True)
    solid_cell = 5
    # SDF negative only at the single solid cell.
    dx = x_c[1] - x_c[0]
    sdf = np.abs(x_c - x_c[solid_cell]) - 0.5 * dx
    assert sdf[solid_cell] < 0 and sdf[solid_cell - 1] > 0 and sdf[solid_cell + 1] > 0

    ibm = construct_ibm(sdf, x_f, rescale=False)
    s = ibm.inside
    # Two eliminations for the single solid cell, both sandwich (sib set, no opp).
    assert s.n_points == 2
    assert np.all(s.row == solid_cell)
    assert np.all(s.sib >= 0)
    assert np.all(s.opp == -1)
    assert np.all(s.coef_w_sib != 0.0)

    # Quadratic exactness of the two-wall reconstruction of the fluid ghosts.
    def quad(x):
        return 2.0 + 0.5 * x - 1.5 * x**2

    u_wall = quad(s.coords[:, 0])
    u_center = quad(x_c[solid_cell])
    for k in range(s.n_points):
        recon = (
            s.coef_c[k] * u_center
            + s.coef_w_self[k] * u_wall[k]
            + s.coef_w_sib[k] * u_wall[s.sib[k]]
        )
        assert recon == pytest.approx(quad(x_c[s.ghost[k]]))


# ---------------------------------------------------------------------------
# Application: matrix / vector / source-matrix
# ---------------------------------------------------------------------------

def _extraction_matrix(ibm, side):
    s = ibm.side(side)
    n = s.n_cells
    A = lil_array((n, n))
    for c, g in zip(s.row, s.ghost):
        A[c, g] += 1.0
    return csr_array(A)


def test_source_matrix_equals_vector():
    x_f, x_c, sdf = _uniform_1d(n=12, x_wall=0.63)
    ibm = construct_ibm(sdf, x_f)
    A = _extraction_matrix(ibm, "outside")
    d = np.array([0.7])  # one IBM point

    _, g_vec = apply_ibm(A, ibm, side="outside", values=d)
    _, G = apply_ibm(A, ibm, side="outside", return_bc="matrix")
    assert np.allclose(G @ d, g_vec)


def test_apply_vector_row_scaling_consistency():
    # For a constant matrix, scaling an independent RHS with apply_ibm_vector
    # must equal the row scaling folded into the modified matrix.
    x_f, x_c, sdf = _uniform_1d(n=12, x_wall=0.63)
    ibm = construct_ibm(sdf, x_f, rescale=True)
    s = ibm.outside
    A = _extraction_matrix(ibm, "outside")
    A_mod, _ = apply_ibm(A, ibm, side="outside")

    b = np.ones(s.n_cells)
    b_scaled = apply_ibm_vector(b, ibm, side="outside")
    # The scaled RHS on cut rows equals row_scale; unchanged elsewhere.
    assert b_scaled[s.row[0]] == pytest.approx(s.row_scale[s.row[0]])
    assert b_scaled[0] == pytest.approx(1.0)


def test_two_sided_independent_values():
    x_f, x_c, sdf = _uniform_1d(n=10, x_wall=0.63)
    ibm = construct_ibm(sdf, x_f)
    A_out = _extraction_matrix(ibm, "outside")
    A_in = _extraction_matrix(ibm, "inside")

    _, g_out = apply_ibm(A_out, ibm, side="outside", values=np.array([1.0]))
    _, g_in = apply_ibm(A_in, ibm, side="inside", values=np.array([9.0]))
    # Sources land on the respective cut cells and differ.
    assert g_out[ibm.outside.row[0]] != 0.0
    assert g_in[ibm.inside.row[0]] != 0.0
    assert g_out[ibm.outside.row[0]] != g_in[ibm.inside.row[0]]


# ---------------------------------------------------------------------------
# End-to-end: immersed Dirichlet diffusion reproduces a linear field exactly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rescale", [True, False])
def test_linear_diffusion_exact(rescale):
    n = 20
    x_f = np.linspace(0.0, 1.0, n + 1)
    _, x_c = generate_grid(n, x_f, generate_x_c=True)
    x_wall = 0.63
    sdf = x_wall - x_c
    ibm = construct_ibm(sdf, x_f, rescale=rescale)
    s = ibm.outside
    cut = int(s.row[0])

    def exact(x):
        return 2.0 + 3.0 * x

    # Assemble: cell 0 pinned, fluid interior Laplacian, solid rows identity.
    A = lil_array((n, n))
    b = np.zeros(n)
    A[0, 0] = 1.0
    b[0] = exact(x_c[0])
    for i in range(1, cut + 1):
        A[i, i - 1] = 1.0
        A[i, i] = -2.0
        A[i, i + 1] = 1.0
    for i in range(cut + 1, n):
        A[i, i] = 1.0
    A = csr_array(A)

    u_wall = exact(s.coords[:, 0])
    A_mod, bc = apply_ibm(A, ibm, side="outside", values=u_wall)
    b_mod = apply_ibm_vector(b, ibm, side="outside")
    u = spsolve(A_mod.tocsc(), b_mod - bc)

    fluid = np.arange(cut + 1)
    assert np.allclose(u[fluid], exact(x_c[fluid]), atol=1e-9)


# ---------------------------------------------------------------------------
# 2-D smoke test
# ---------------------------------------------------------------------------

def test_2d_construct_and_apply():
    nx, ny = 12, 10
    x_f = np.linspace(0.0, 1.0, nx + 1)
    y_f = np.linspace(0.0, 1.0, ny + 1)
    _, x_c = generate_grid(nx, x_f, generate_x_c=True)
    _, y_c = generate_grid(ny, y_f, generate_x_c=True)
    X, Y = np.meshgrid(x_c, y_c, indexing="ij")
    # Circular solid.
    sdf = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) - 0.25

    ibm = construct_ibm(sdf, [x_f, y_f])
    assert ibm.outside.n_points > 0
    assert ibm.inside.n_points > 0
    assert ibm.outside.coords.shape[1] == 2

    n = sdf.size
    A = csr_array(np.eye(n))
    A_mod, G = apply_ibm(A, ibm, side="outside", return_bc="matrix")
    assert A_mod.shape == (n, n)
    assert G.shape == (n, ibm.outside.n_points)
