"""Tests for the directional ghost-cell IBM (:mod:`pymrm.ibm`)."""

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
    """Uniform 1-D grid on [0, 1] with a planar wall at *x_wall*."""
    x_f = np.linspace(0.0, 1.0, n + 1)
    _, x_c = generate_grid(n, x_f, generate_x_c=True)
    sdf = x_wall - x_c
    return x_c, sdf


def _ref_point_value(theta):
    d = theta * (1.0 + theta)
    return 2.0 * (theta**2 - 1.0) / d, theta * (1.0 - theta) / d, 2.0 / d


def _extraction_csr(n, rows, cols):
    A = lil_array((n, n))
    for r, c in zip(rows, cols):
        A[r, c] += 1.0
    return csr_array(A)


# ---------------------------------------------------------------------------
# Geometry and classification
# ---------------------------------------------------------------------------

def test_classification_basic():
    x_c, sdf = _uniform_1d(n=10, x_wall=0.63)
    ibm = construct_ibm(sdf, x_c)

    assert ibm.n_crossings == 1
    assert ibm.row_out[0] == 5
    assert ibm.ghost_out[0] == 6
    assert ibm.opp_out[0] == 4
    assert ibm.direction[0] == 1
    assert ibm.row_in[0] == 6
    assert ibm.ghost_in[0] == 5
    # Pure spatial: no non-spatial dims
    assert ibm.ns_size == 1
    assert ibm.n_cells == ibm.n_spatial_cells


def test_crossing_keys_unique():
    x_c, sdf = _uniform_1d(n=10, x_wall=0.63)
    ibm = construct_ibm(sdf, x_c)
    assert ibm.crossing_key.size == np.unique(ibm.crossing_key).size


def test_coords_on_wall():
    x_c, sdf = _uniform_1d(n=10, x_wall=0.63)
    ibm = construct_ibm(sdf, x_c)
    assert 0.55 < ibm.coords[0, 0] < 0.65


# ---------------------------------------------------------------------------
# Lagrange coefficient correctness
# ---------------------------------------------------------------------------

def test_equidistant_reduction_matches_reference():
    x_c, sdf = _uniform_1d(n=10, x_wall=0.63)
    ibm = construct_ibm(sdf, x_c, rescale=False)
    theta = (0.63 - x_c[5]) / (x_c[6] - x_c[5])
    c_center, c_opp, c_wall = _ref_point_value(theta)
    assert ibm.coef_c_out[0] == pytest.approx(c_center)
    assert ibm.coef_o_out[0] == pytest.approx(c_opp)
    assert ibm.coef_w_out[0] == pytest.approx(c_wall)


def test_reconstruction_second_order_convergence():
    def field(x):
        return np.exp(1.3 * x)

    errors = []
    for n in (20, 40, 80, 160):
        x_f = np.linspace(0.0, 1.0, n + 1)
        _, x_c = generate_grid(n, x_f, generate_x_c=True)
        k = n // 2
        x_wall = x_c[k] + 0.5 * (x_c[k + 1] - x_c[k])
        sdf = x_wall - x_c
        ibm = construct_ibm(sdf, x_c, rescale=False)

        u = field(x_c)
        u_wall = field(ibm.coords[:, 0])
        A = _extraction_csr(x_c.size, ibm.row_out, ibm.ghost_out)
        A_mod, bc = apply_ibm(A, ibm, values_outside=u_wall,
                               values_inside=np.zeros(ibm.n_crossings))
        recon = (A_mod @ u) + bc
        errors.append(abs(recon[k] - field(x_c[k + 1])))

    orders = np.log(np.array(errors[:-1]) / np.array(errors[1:])) / np.log(2.0)
    assert np.all(orders > 2.5)


def test_nonuniform_quadratic_exact():
    n = 15
    x_f = np.sort(np.concatenate(
        ([0.0, 1.0], np.random.default_rng(0).uniform(0, 1, n - 1))
    ))
    _, x_c = generate_grid(n, x_f, generate_x_c=True)
    sdf = 0.6123 - x_c
    ibm = construct_ibm(sdf, x_c, rescale=False)

    def quad(x):
        return 1.0 - 2.0 * x + 3.0 * x**2

    u = quad(x_c)
    u_wall = quad(ibm.coords[:, 0])
    opp_safe = np.where(ibm.opp_out >= 0, ibm.opp_out, 0)
    recon = (
        ibm.coef_c_out * u[ibm.row_out]
        + ibm.coef_o_out * np.where(ibm.opp_out >= 0, u[opp_safe], 0.0)
        + ibm.coef_w_out * u_wall
    )
    assert np.allclose(recon, quad(x_c[ibm.ghost_out]))


# ---------------------------------------------------------------------------
# Sandwich (thin solid)
# ---------------------------------------------------------------------------

def test_sandwich_structure_and_exactness():
    n = 11
    x_f = np.linspace(0.0, 1.0, n + 1)
    _, x_c = generate_grid(n, x_f, generate_x_c=True)
    solid_cell = 5
    dx = x_c[1] - x_c[0]
    sdf = np.abs(x_c - x_c[solid_cell]) - 0.5 * dx
    assert sdf[solid_cell] < 0 and sdf[solid_cell - 1] > 0 and sdf[solid_cell + 1] > 0

    ibm = construct_ibm(sdf, x_c, rescale=False)

    assert ibm.n_crossings == 2
    assert np.all(ibm.row_in == solid_cell)
    assert np.all(ibm.sib_in >= 0)
    assert np.all(ibm.opp_in == -1)
    assert np.all(ibm.coef_w_sib_in != 0.0)
    assert np.all(ibm.sib_out == -1)

    def quad(x):
        return 2.0 + 0.5 * x - 1.5 * x**2

    u_wall = quad(ibm.coords[:, 0])
    u_center = quad(x_c[solid_cell])
    for k in range(ibm.n_crossings):
        recon = (
            ibm.coef_c_in[k] * u_center
            + ibm.coef_w_in[k] * u_wall[k]
            + ibm.coef_w_sib_in[k] * u_wall[ibm.sib_in[k]]
        )
        assert recon == pytest.approx(quad(x_c[ibm.ghost_in[k]]))


# ---------------------------------------------------------------------------
# apply_ibm: source vector / matrix / vector consistency
# ---------------------------------------------------------------------------

def test_source_matrix_equals_vector():
    x_c, sdf = _uniform_1d(n=12, x_wall=0.63)
    ibm = construct_ibm(sdf, x_c)
    A = _extraction_csr(x_c.size, ibm.row_out, ibm.ghost_out)

    d = np.full(ibm.n_crossings, 0.7)
    _, g_vec = apply_ibm(A, ibm, values_outside=d,
                          values_inside=np.zeros(ibm.n_crossings))
    _, G_out, G_in = apply_ibm(A, ibm, return_bc="matrix")
    assert np.allclose(np.asarray(G_out @ d).ravel(), g_vec)


def test_apply_ibm_vector_consistency():
    x_c, sdf = _uniform_1d(n=12, x_wall=0.63)
    ibm = construct_ibm(sdf, x_c, rescale=True)
    b = np.ones(ibm.n_cells)
    b_scaled = apply_ibm_vector(b, ibm)
    cut_out = ibm.row_out[0]
    assert b_scaled[cut_out] == pytest.approx(ibm.row_scale_out[cut_out])
    assert b_scaled[0] == pytest.approx(1.0)


def test_values_none_defaults():
    x_c, sdf = _uniform_1d(n=10, x_wall=0.63)
    ibm = construct_ibm(sdf, x_c, rescale=False)
    A = csr_array(np.eye(ibm.n_cells))

    _, g_zero = apply_ibm(A, ibm)
    assert np.all(g_zero == 0.0)

    d = np.full(ibm.n_crossings, 3.0)
    _, g_shared_out = apply_ibm(A, ibm, values_outside=d)
    _, g_shared_in = apply_ibm(A, ibm, values_inside=d)
    _, g_both = apply_ibm(A, ibm, values_outside=d, values_inside=d)
    assert np.allclose(g_shared_out, g_both)
    assert np.allclose(g_shared_in, g_both)


def test_two_sided_independent_values():
    x_c, sdf = _uniform_1d(n=10, x_wall=0.63)
    ibm = construct_ibm(sdf, x_c, rescale=False)
    n = ibm.n_cells
    A_out = _extraction_csr(n, ibm.row_out, ibm.ghost_out)
    A_in = _extraction_csr(n, ibm.row_in, ibm.ghost_in)
    A = (csr_array(A_out) + csr_array(A_in)).tocsr()

    _, g = apply_ibm(A, ibm, values_outside=np.array([1.0]),
                      values_inside=np.array([9.0]))
    assert g[ibm.row_out[0]] != 0.0
    assert g[ibm.row_in[0]] != 0.0
    assert g[ibm.row_out[0]] != g[ibm.row_in[0]]


# ---------------------------------------------------------------------------
# Non-spatial axes (ns dimensions)
# ---------------------------------------------------------------------------

def test_axes_and_shape_pure_spatial():
    """Explicitly specifying axes for a pure spatial case gives same result."""
    x_c, sdf = _uniform_1d(n=10, x_wall=0.63)
    ibm_default = construct_ibm(sdf, x_c)
    ibm_explicit = construct_ibm(sdf, x_c, axes=(0,), shape=(sdf.size,))
    assert ibm_default.n_crossings == ibm_explicit.n_crossings
    assert ibm_default.ns_size == 1
    assert ibm_explicit.ns_size == 1


def test_ns_dimension_block_structure():
    """With a component axis, each IBM crossing spans Nc rows in the full grid."""
    n, Nc = 12, 3
    x_f = np.linspace(0.0, 1.0, n + 1)
    _, x_c = generate_grid(n, x_f, generate_x_c=True)
    x_wall = 0.63
    sdf = x_wall - x_c

    ibm = construct_ibm(sdf, x_c, axes=(0,), shape=(n, Nc))
    assert ibm.ns_size == Nc
    assert ibm.n_cells == n * Nc
    assert ibm.n_spatial_cells == n

    # For a block-diagonal matrix (Nc decoupled diffusion problems), the source
    # vector for per-component wall values should equal Nc independent 1D IBM applies.
    A_blk = csr_array(np.eye(n * Nc))
    val_out_ns = np.array([[1.0, 2.0, 3.0]])  # shape (1 crossing, 3 components)
    _, g_ns = apply_ibm(A_blk, ibm, values_outside=val_out_ns,
                         values_inside=np.zeros((1, Nc)))

    # Reference: construct a separate pure 1D IBM and apply for each component
    ibm_1d = construct_ibm(sdf, x_c)  # pure spatial
    A_1d = csr_array(np.eye(n))
    for c in range(Nc):
        _, g_1d = apply_ibm(A_1d, ibm_1d, values_outside=np.array([val_out_ns[0, c]]),
                              values_inside=np.zeros(1))
        # Pick rows corresponding to component c: full_row[k, c] = row_out[k] * Nc + c
        full_row_k_c = int(ibm.row_out[0]) * Nc + c
        assert g_ns[full_row_k_c] == pytest.approx(g_1d[ibm_1d.row_out[0]])


def test_ns_source_matrix_shape():
    n, Nc = 10, 4
    x_f = np.linspace(0.0, 1.0, n + 1)
    _, x_c = generate_grid(n, x_f, generate_x_c=True)
    sdf = 0.63 - x_c
    ibm = construct_ibm(sdf, x_c, axes=(0,), shape=(n, Nc))

    A = csr_array(np.eye(ibm.n_cells))
    _, G_out, G_in = apply_ibm(A, ibm, return_bc="matrix")
    assert G_out.shape == (ibm.n_cells, ibm.n_crossings * Nc)
    assert G_in.shape == (ibm.n_cells, ibm.n_crossings * Nc)

    # Source via matrix equals source via vector
    vals = np.random.default_rng(0).random((ibm.n_crossings, Nc))
    _, g_vec = apply_ibm(A, ibm, values_outside=vals, values_inside=np.zeros_like(vals))
    g_mat = np.asarray(G_out @ vals.ravel()).ravel()
    assert np.allclose(g_mat, g_vec)


# ---------------------------------------------------------------------------
# End-to-end: 1-D immersed Dirichlet diffusion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rescale", [True, False])
def test_linear_diffusion_exact(rescale):
    n = 20
    x_f = np.linspace(0.0, 1.0, n + 1)
    _, x_c = generate_grid(n, x_f, generate_x_c=True)
    sdf = 0.63 - x_c
    ibm = construct_ibm(sdf, x_c, rescale=rescale)
    cut = int(ibm.row_out[0])

    def exact(x):
        return 2.0 + 3.0 * x

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

    u_wall = exact(ibm.coords[:, 0])
    A_mod, bc = apply_ibm(A, ibm, values_outside=u_wall,
                           values_inside=np.zeros(ibm.n_crossings))
    b_mod = apply_ibm_vector(b, ibm)
    u = spsolve(A_mod.tocsc(), b_mod - bc)

    assert np.allclose(u[:cut + 1], exact(x_c[:cut + 1]), atol=1e-9)


# ---------------------------------------------------------------------------
# 2-D smoke test with explicit axes
# ---------------------------------------------------------------------------

def test_2d_construct_and_apply():
    nx, ny = 12, 10
    x_f0 = np.linspace(0.0, 1.0, nx + 1)
    x_f1 = np.linspace(0.0, 1.0, ny + 1)
    _, x_c0 = generate_grid(nx, x_f0, generate_x_c=True)
    _, x_c1 = generate_grid(ny, x_f1, generate_x_c=True)
    X, Y = np.meshgrid(x_c0, x_c1, indexing="ij")
    sdf = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) - 0.25

    ibm = construct_ibm(sdf, [x_c0, x_c1], axes=(0, 1))
    assert ibm.n_crossings > 0
    assert ibm.coords.shape == (ibm.n_crossings, 2)
    assert ibm.ns_size == 1
    assert ibm.n_cells == nx * ny

    n = ibm.n_cells
    A = csr_array(np.eye(n))
    A_mod, G_out, G_in = apply_ibm(A, ibm, return_bc="matrix")
    assert A_mod.shape == (n, n)
    assert G_out.shape == (n, ibm.n_crossings)
    assert G_in.shape == (n, ibm.n_crossings)


def test_2d_with_components():
    """2-D spatial + 1 component axis: shape (Nx, Ny, Nc) with axes=(0, 1)."""
    nx, ny, Nc = 8, 7, 2
    _, x_c0 = generate_grid(nx, [0.0, 1.0], generate_x_c=True)
    _, x_c1 = generate_grid(ny, [0.0, 1.0], generate_x_c=True)
    X, Y = np.meshgrid(x_c0, x_c1, indexing="ij")
    sdf = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) - 0.25

    ibm = construct_ibm(sdf, [x_c0, x_c1], axes=(0, 1), shape=(nx, ny, Nc))
    assert ibm.ns_size == Nc
    assert ibm.n_cells == nx * ny * Nc
    assert ibm.n_spatial_cells == nx * ny
    assert ibm.ns_shape == (Nc,)

    n = ibm.n_cells
    A = csr_array(np.eye(n))
    A_mod, g = apply_ibm(A, ibm, values_outside=np.ones((ibm.n_crossings, Nc)))
    assert A_mod.shape == (n, n)
    assert g.shape == (n,)
