"""Tests for the directional ghost-cell IBM (:mod:`pymrm.ibm`)."""

import numpy as np
import pytest
from scipy.sparse import csr_array, lil_array, coo_array
from scipy.sparse.linalg import spsolve

from pymrm.ibm import construct_ibm, apply_ibm, apply_ibm_vector, _expand_full
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


def _circle_2d(n=12, radius=0.25, center=(0.5, 0.5)):
    """2-D uniform grid on [0,1]^2 with a solid disk (sdf < 0 inside)."""
    x_f = np.linspace(0.0, 1.0, n + 1)
    _, x_c = generate_grid(n, x_f, generate_x_c=True)
    xx, yy = np.meshgrid(x_c, x_c, indexing="ij")
    sdf = np.hypot(xx - center[0], yy - center[1]) - radius
    return [x_c, x_c], sdf


def _two_slabs_1d(n=20):
    """1-D grid with two solid slabs -> four wall crossings."""
    x_f = np.linspace(0.0, 1.0, n + 1)
    _, x_c = generate_grid(n, x_f, generate_x_c=True)
    sdf = np.minimum(np.abs(x_c - 0.275) - 0.075, np.abs(x_c - 0.70) - 0.10)
    return x_c, sdf


def _ghost_extraction_full(ibm, value=1.0):
    """CSR with *value* at every (full_row, full_ghost) ghost position.

    Gives ``apply_ibm`` a nonzero ghost entry to fold on every crossing and
    non-spatial layer, so the resulting source vector is nonzero and identical
    across layers (each layer decouples into a copy of the spatial problem).
    """
    fr = np.concatenate([_expand_full(ibm, ibm.row_out).ravel(),
                         _expand_full(ibm, ibm.row_in).ravel()])
    fg = np.concatenate([_expand_full(ibm, ibm.ghost_out).ravel(),
                         _expand_full(ibm, ibm.ghost_in).ravel()])
    data = np.full(fr.size, float(value))
    return coo_array((data, (fr, fg)),
                     shape=(ibm.n_cells, ibm.n_cells)).tocsr()


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
# Multi-dimensional point-value broadcasting (strict numpy semantics)
# ---------------------------------------------------------------------------

def _interleaved_ibm(n=12, npn=2, ncn=3):
    """IBM on an interleaved field (nx, np, ny, nc), spatial axes (0, 2)."""
    x_c, sdf = _circle_2d(n=n)
    ibm = construct_ibm(sdf, x_c, axes=(0, 2), shape=(n, npn, n, ncn),
                        rescale=False)
    return ibm


@pytest.mark.parametrize("make_val", [
    lambda npnt, npn, ncn, rng: 3.0,
    lambda npnt, npn, ncn, rng: rng.random(ncn),            # (nc,)
    lambda npnt, npn, ncn, rng: rng.random((npn, 1)),       # (np, 1)
    lambda npnt, npn, ncn, rng: rng.random((npn, ncn)),     # (np, nc)
    lambda npnt, npn, ncn, rng: rng.random((npnt, 1, 1)),   # per crossing
    lambda npnt, npn, ncn, rng: rng.random((npnt, npn, ncn)),  # full
])
def test_broadcast_shapes_match_explicit(make_val):
    """Any accepted shape equals its explicit (npnt, np, nc) broadcast."""
    npn, ncn = 2, 3
    ibm = _interleaved_ibm(n=12, npn=npn, ncn=ncn)
    A = _ghost_extraction_full(ibm)
    npnt = ibm.n_crossings
    rng = np.random.default_rng(3)
    val = make_val(npnt, npn, ncn, rng)
    explicit = np.broadcast_to(np.asarray(val, dtype=float),
                               (npnt, npn, ncn)).copy()
    _, g1 = apply_ibm(A, ibm, values_outside=val)
    _, g2 = apply_ibm(A, ibm, values_outside=explicit)
    assert np.allclose(g1, g2)


def test_broadcast_layer_ordering():
    """Canonical (np, nc) lands each value on the correct (p, c) layer rows.

    With a constant ghost matrix each non-spatial layer is an independent copy
    of the spatial problem, so the source at spatial cell *s*, layer *j* must be
    ``val_flat[j] * g_unit[s]`` with ``j`` the C-order index over (np, nc).
    """
    npn, ncn = 2, 3
    x_c, sdf = _circle_2d(n=12)
    ibm_s = construct_ibm(sdf, x_c, rescale=False)          # pure spatial
    A_s = _ghost_extraction_full(ibm_s)
    _, g_unit = apply_ibm(A_s, ibm_s, values_outside=1.0, values_inside=1.0)

    ibm = construct_ibm(sdf, x_c, axes=(0, 2), shape=(12, npn, 12, ncn),
                        rescale=False)
    A = _ghost_extraction_full(ibm)
    rng = np.random.default_rng(5)
    val = rng.random((npn, ncn))
    _, g_ns = apply_ibm(A, ibm, values_outside=val, values_inside=val)

    frows = _expand_full(ibm, np.arange(ibm.n_spatial_cells))   # (n_spatial, ns)
    g_grid = g_ns[frows]                                        # (n_spatial, ns)
    expected = g_unit[:, None] * val.reshape(-1)[None, :]
    assert np.allclose(g_grid, expected)


def test_broadcast_bare_1d_with_ns_axes_errors():
    """A bare (npnt,) array is rejected (with a hint) when ns axes are present."""
    ibm = _interleaved_ibm(n=12, npn=2, ncn=3)
    A = _ghost_extraction_full(ibm)
    npnt = ibm.n_crossings
    assert npnt != 3          # otherwise it would broadcast onto the nc axis
    with pytest.raises(ValueError, match="per-crossing"):
        apply_ibm(A, ibm, values_outside=np.ones(npnt))
    with pytest.raises(ValueError, match="broadcastable"):
        apply_ibm(A, ibm, values_outside=np.ones((npnt, 7)))


def test_broadcast_collision_uses_numpy_semantics():
    """When nc == n_crossings a 1-D array follows numpy (per last axis)."""
    x_c, sdf = _two_slabs_1d(n=20)
    ibm0 = construct_ibm(sdf, x_c)
    npnt = ibm0.n_crossings
    assert npnt >= 2
    ibm = construct_ibm(sdf, x_c, axes=(0,), shape=(20, npnt), rescale=False)
    A = _ghost_extraction_full(ibm)
    v = np.arange(1.0, npnt + 1.0)                    # (npnt,) == (ncn,)

    _, g_bare = apply_ibm(A, ibm, values_outside=v)
    _, g_percomp = apply_ibm(A, ibm, values_outside=v.reshape(1, npnt))
    _, g_percross = apply_ibm(A, ibm, values_outside=v.reshape(npnt, 1))
    assert np.allclose(g_bare, g_percomp)             # numpy: aligns to last axis
    assert not np.allclose(g_bare, g_percross)        # NOT per-crossing


def test_pure_spatial_value_backcompat():
    """Pure-spatial fields keep the scalar / (npnt,) per-crossing behaviour."""
    x_c, sdf = _uniform_1d(n=10, x_wall=0.63)
    ibm = construct_ibm(sdf, x_c, rescale=False)       # ns_shape == ()
    A = _ghost_extraction_full(ibm)
    _, g_scalar = apply_ibm(A, ibm, values_outside=2.0)
    _, g_1d = apply_ibm(A, ibm, values_outside=np.full(ibm.n_crossings, 2.0))
    assert np.allclose(g_scalar, g_1d)


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


# ---------------------------------------------------------------------------
# Solid adjacent to the domain boundary (first-order fallback)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("solid_cell", [-1, 0])
def test_boundary_adjacent_solid_fallback(solid_cell):
    """A cut cell whose opposite neighbour is outside the domain must use the
    first-order fallback instead of indexing past the grid edge.

    With the solid pressed against a domain wall, the inside reconstruction has
    no opposite same-side cell (``opp == -1``); it must fall back to a two-point
    (cut cell + wall) formula.  Regression test for an out-of-bounds gather of
    the opposite-cell coordinate at the upper boundary.
    """
    n = 10
    x_f = np.linspace(0.0, 1.0, n + 1)
    _, x_c = generate_grid(n, x_f, generate_x_c=True)
    sdf = np.ones(n)
    sdf[solid_cell] = -1.0                       # single solid cell at a wall

    with pytest.warns(RuntimeWarning, match="first-order"):
        ibm = construct_ibm(sdf, x_c)

    assert ibm.n_crossings == 1
    k = 0
    assert ibm.opp_in[k] == -1                   # no opposite same-side cell
    assert ibm.coef_o_in[k] == 0.0
    assert ibm.coef_w_sib_in[k] == 0.0

    # The two-point fallback reconstructs the fluid ghost value from the solid
    # cut cell and the wall value; it is exact for a linear field.
    a, b = 0.4, -1.3
    u = a + b * x_c
    u_wall = a + b * ibm.coords[k, 0]
    recon = ibm.coef_c_in[k] * u[ibm.row_in[k]] + ibm.coef_w_in[k] * u_wall
    assert recon == pytest.approx(u[ibm.ghost_in[k]])


def test_boundary_adjacent_solid_2d_no_crash():
    """A solid touching a domain wall (cut cells in the outer layer) builds."""
    nx, ny = 20, 20
    _, x_c0 = generate_grid(nx, np.linspace(0.0, 1.0, nx + 1), generate_x_c=True)
    _, x_c1 = generate_grid(ny, np.linspace(0.0, 1.0, ny + 1), generate_x_c=True)
    # one-cell-thick solid strip pressed against the bottom wall (row 0): its
    # inside reconstruction has the fluid ghost inward and no cell outward.
    sdf = np.ones((nx, ny))
    sdf[8:12, 0] = -1.0

    with pytest.warns(RuntimeWarning, match="first-order"):
        ibm = construct_ibm(sdf, [x_c0, x_c1], axes=(0, 1))

    assert ibm.n_crossings > 0
    assert np.any(ibm.opp_in == -1)              # inside fallback exercised
    n = ibm.n_cells
    A_mod, g = apply_ibm(csr_array(np.eye(n)), ibm,
                         values_inside=np.zeros(ibm.n_crossings))
    assert np.all(np.isfinite(A_mod.toarray()))
    assert np.all(np.isfinite(g))


# ---------------------------------------------------------------------------
# reconstruct_ghost_values / fill_ghost_values
# ---------------------------------------------------------------------------

from pymrm.ibm import reconstruct_ghost_values, fill_ghost_values


def test_ghost_values_quadratic_exact_1d():
    # second-order Lagrange (opp, row, wall) reproduces quadratics exactly
    x_c, sdf = _uniform_1d(n=10, x_wall=0.63)

    def f(x):
        return 1.0 + 2.0 * x + 3.0 * x**2

    ibm = construct_ibm(sdf, x_c)
    u = f(x_c)
    u[sdf <= 0.0] = 0.0  # solid values must not be used
    w = f(ibm.coords[:, 0])
    ghosts, vals = reconstruct_ghost_values(ibm, x_c, u, wall_values=w)
    assert np.allclose(vals[:, ], f(x_c[ghosts]))


def test_ghost_values_theta_switch_first_order():
    # wall extremely close to the cut-cell center: second-order Lagrange
    # would blow up; the first-order (opp, wall) reconstruction stays
    # bounded and is exact for linear fields
    n = 10
    x_f = np.linspace(0.0, 1.0, n + 1)
    _, x_c = generate_grid(n, x_f, generate_x_c=True)
    x_wall = x_c[5] + 1.0e-3 * (x_c[6] - x_c[5])
    sdf = x_wall - x_c

    def f(x):
        return 0.3 + 1.7 * x

    ibm = construct_ibm(sdf, x_c)
    u = f(x_c)
    u[sdf <= 0.0] = 0.0
    w = f(ibm.coords[:, 0])
    ghosts, vals = reconstruct_ghost_values(ibm, x_c, u, wall_values=w,
                                            theta_min=0.25)
    assert np.allclose(vals[:, ], f(x_c[ghosts]))
    # bounded also for a strongly curved field (no 1/theta blow-up)
    u2 = np.exp(3.0 * x_c)
    u2[sdf <= 0.0] = 0.0
    _, vals2 = reconstruct_ghost_values(ibm, x_c, u2,
                                        wall_values=np.exp(3.0 * x_wall),
                                        theta_min=0.25)
    assert np.all(np.abs(vals2) < 10.0 * np.exp(3.0))


def test_ghost_values_no_opp_linear():
    # cut cell at the domain edge has no opposite neighbour: linear
    # (row, wall) reconstruction, exact for linear fields
    n = 4
    x_f = np.linspace(0.0, 1.0, n + 1)
    _, x_c = generate_grid(n, x_f, generate_x_c=True)
    x_wall = 0.30  # fluid region = first cell only -> no opp
    sdf = x_wall - x_c

    def f(x):
        return 2.0 - 0.8 * x

    with pytest.warns(RuntimeWarning, match="domain boundary"):
        ibm = construct_ibm(sdf, x_c)
    assert (ibm.opp_out < 0).any()
    u = f(x_c)
    u[sdf <= 0.0] = 0.0
    w = f(ibm.coords[:, 0])
    ghosts, vals = reconstruct_ghost_values(ibm, x_c, u, wall_values=w)
    assert np.allclose(vals[:, ], f(x_c[ghosts]))


def test_fill_ghost_values_2d_circle():
    # 2-D disc: filled ghost ring approximates a smooth field near the wall
    n = 40
    x_f = np.linspace(-1.0, 1.0, n + 1)
    _, x_c = generate_grid(n, x_f, generate_x_c=True)
    XX, YY = np.meshgrid(x_c, x_c, indexing="ij")
    r = np.sqrt(XX**2 + YY**2)
    sdf = 0.75 - r  # fluid inside the disc

    def f(x, y):
        return 1.0 + 0.5 * x - 0.3 * y

    ibm = construct_ibm(sdf, [x_c, x_c])
    u = f(XX, YY)
    u[sdf <= 0.0] = 0.0
    # wall values from the exact field at the crossing coordinates
    w = f(ibm.coords[:, 0], ibm.coords[:, 1])
    filled = fill_ghost_values(ibm, [x_c, x_c], u, wall_values=w)
    ghosts = np.unique(ibm.ghost_out)
    err = np.abs(filled.ravel()[ghosts]
                 - f(XX.ravel()[ghosts], YY.ravel()[ghosts]))
    # direction-averaged reconstructions of a linear field on a curved wall
    # are first-order consistent; on this grid the error is well below h
    assert err.max() < 0.05
    # untouched cells unchanged
    untouched = np.setdiff1d(np.arange(u.size), ghosts)
    assert np.array_equal(filled.ravel()[untouched], u.ravel()[untouched])


def test_ghost_values_multicomponent_trailing_axis():
    x_c, sdf = _uniform_1d(n=10, x_wall=0.63)
    ibm = construct_ibm(sdf, x_c)
    u = np.stack([1.0 + 2.0 * x_c, 3.0 - x_c], axis=-1)  # (n, 2)
    u[sdf <= 0.0] = 0.0
    w = np.stack([1.0 + 2.0 * ibm.coords[:, 0], 3.0 - ibm.coords[:, 0]],
                 axis=-1)
    ghosts, vals = reconstruct_ghost_values(ibm, x_c, u, wall_values=w)
    expect = np.stack([1.0 + 2.0 * x_c[ghosts], 3.0 - x_c[ghosts]], axis=-1)
    assert vals.shape == (ibm.n_crossings, 2)
    assert np.allclose(vals, expect)
