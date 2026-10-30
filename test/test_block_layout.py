"""Tests for BlockLayout: named domains concatenated into one state."""

import numpy as np
import pytest
from scipy.sparse import coo_array, csc_array, csr_array, dia_array, eye_array

from pymrm import (BlockLayout, construct_coefficient_matrix, construct_convflux_upwind,
                   construct_div, construct_grad, construct_boundary_value_matrices,
                   generate_grid, update_array_indices)

RNG = np.random.default_rng(1234)


def _parts_along(axis, base=(3, 4, 2), lengths=(2, 1, 3)):
    """Random parts that differ only along `axis`, keyed by name."""
    parts = {}
    for k, length in enumerate(lengths):
        shape = list(base)
        shape[axis] = length
        parts[f"d{k}"] = RNG.standard_normal(shape)
    return parts


def _layouts():
    """A layout per concatenation axis, and a region-major one."""
    out = []
    for axis in (0, 1, -1):
        parts = _parts_along(axis)
        out.append(BlockLayout({k: v.shape for k, v in parts.items()}, axis=axis))
    out.append(BlockLayout({"a": (3, 2, 2), "b": (3, 4, 1), "c": (2, 3)}, axis=None))
    return out


# -- 1. definition -------------------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1, -1, 2, None])
def test_view_and_put_reproduce_concatenate(axis):
    if axis is None:
        parts = {"a": RNG.standard_normal((3, 2, 2)), "b": RNG.standard_normal((3, 4, 1)),
                 "c": RNG.standard_normal((5,))}
        expected = np.concatenate([p.ravel() for p in parts.values()])
    else:
        parts = _parts_along(axis)
        expected = np.concatenate(list(parts.values()), axis=axis)
    layout = BlockLayout({k: v.shape for k, v in parts.items()}, axis=axis)
    assert layout.shape == expected.shape
    assert layout.size == expected.size

    u = np.zeros(layout.shape)
    for name, part in parts.items():
        layout.put(u, name, part.ravel())  # any shape with the right size
    np.testing.assert_array_equal(u, expected)

    for name, part in parts.items():
        view = layout.view(u, name)
        assert view.shape == part.shape
        np.testing.assert_array_equal(view, part)
        assert np.shares_memory(view, u)
        view[...] = -1.0  # writes land in u
        np.testing.assert_array_equal(layout.view(u, name), -np.ones(part.shape))
        np.testing.assert_array_equal(u.ravel()[layout.index(name).ravel()], -1.0)

    # a flat state works as well as a shaped one
    flat = expected.ravel().copy()
    for name, part in parts.items():
        np.testing.assert_array_equal(layout.view(flat, name), part)


def test_view_refuses_to_copy():
    layout = BlockLayout({"a": (3, 2), "b": (3, 1)}, axis=1)
    with pytest.raises(ValueError, match="contiguous"):
        layout.view(np.zeros(layout.shape, order="F"), "a")
    with pytest.raises(ValueError, match="contiguous"):
        layout.view(np.zeros(2 * layout.size)[::2], "a")
    with pytest.raises(ValueError, match="contiguous"):
        layout.put(np.zeros(layout.shape, order="F"), "a", np.ones(6))
    with pytest.raises(TypeError):
        layout.view([0.0] * layout.size, "a")
    with pytest.raises(ValueError, match="entries"):
        layout.view(np.zeros(layout.size + 1), "a")
    with pytest.raises(ValueError, match="entries"):
        layout.put(np.zeros(layout.shape), "a", np.ones(5))


def test_index_matches_concatenated_arange():
    for layout in _layouts():
        flat = np.arange(layout.size).reshape(layout.shape)
        for name, shape in layout.shapes.items():
            idx = layout.index(name)
            assert idx.shape == shape
            np.testing.assert_array_equal(idx, layout.view(flat, name))
        # the domains tile the state exactly once
        all_idx = np.sort(np.concatenate([layout.index(n).ravel() for n in layout.shapes]))
        np.testing.assert_array_equal(all_idx, np.arange(layout.size))


def test_pick_extracts_and_scatters():
    for layout in _layouts():
        u = RNG.standard_normal(layout.size)
        for name, shape in layout.shapes.items():
            pick = layout.pick(name)
            assert isinstance(pick, csr_array)
            assert pick.shape == (int(np.prod(shape)), layout.size)
            np.testing.assert_array_equal(pick @ u, layout.view(u, name).ravel())
            v = RNG.standard_normal(pick.shape[0])
            scattered = np.zeros(layout.size)
            layout.put(scattered, name, v)
            np.testing.assert_array_equal(pick.T @ v, scattered)


# -- 2. place against pick -------------------------------------------------------


def _random_block(n_rows, n_cols, fmt):
    dense = RNG.standard_normal((n_rows, n_cols)) * (RNG.random((n_rows, n_cols)) < 0.4)
    if fmt == "dia":  # a few diagonals, as DIA is meant for
        offsets = np.subtract.outer(np.arange(n_rows), np.arange(n_cols))
        dense[np.abs(offsets) > 3] = 0.0
    return {"csr": csr_array, "csc": csc_array, "dia": dia_array, "coo": coo_array}[fmt](dense)


@pytest.mark.parametrize("fmt", ["csr", "csc", "dia", "coo"])
def test_place_equals_pick_sandwich(fmt):
    for layout in _layouts():
        names = list(layout.shapes) + [None]
        sizes = {n: int(np.prod(s)) for n, s in layout.shapes.items()}
        for a in names:
            for b in names:
                # None: a side that already spans the state, or a local side of any size
                for none_size in (layout.size, 3):
                    n_r = sizes[a] if a is not None else none_size
                    n_c = sizes[b] if b is not None else none_size
                    block = _random_block(n_r, n_c, fmt)
                    left = layout.pick(a).T if a is not None else eye_array(n_r)
                    right = layout.pick(b) if b is not None else eye_array(n_c)
                    expected = (left @ block @ right).toarray()
                    placed = layout.place(block, a, b)
                    assert isinstance(placed, csr_array if fmt == "csr" else csc_array)
                    np.testing.assert_array_equal(placed.toarray(), expected)


def test_place_identity_is_pick():
    layout = _layouts()[1]
    for name, shape in layout.shapes.items():
        n = int(np.prod(shape))
        np.testing.assert_array_equal(layout.place(eye_array(n, format="csr"), None, name).toarray(),
                                      layout.pick(name).toarray())


# -- 3. guards -------------------------------------------------------------------


def test_place_rejects_wrong_block_sizes():
    # update_array_indices itself accepts these silently
    layout = BlockLayout({"a": (2, 2), "b": (2, 2)}, axis=0)
    for shape in [(4, 3), (3, 4), (5, 4), (4, 5)]:
        with pytest.raises(ValueError, match="domain 'a'"):
            layout.place(csc_array(np.ones(shape)), "a", "a")
    with pytest.raises(ValueError, match="domain 'b'"):
        layout.place(csr_array(np.ones((4, 3))), None, "b")


def test_unknown_names_raise():
    layout = BlockLayout({"a": (2, 1), "b": (2, 1)}, axis=-1)
    u = np.zeros(layout.shape)
    for call in (lambda: layout.index("c"), lambda: layout.view(u, "c"),
                 lambda: layout.put(u, "c", 0.0), lambda: layout.pick("c"),
                 lambda: layout.place(eye_array(2), "a", "c"),
                 lambda: layout.assemble([(eye_array(2), "c", "a")])):
        with pytest.raises(KeyError, match="unknown domain"):
            call()


def test_invalid_layouts_raise():
    with pytest.raises(ValueError, match="axis=None"):
        BlockLayout({"a": (3, 2, 1), "b": (3, 2, 2)}, axis=1)  # unequal trailing size
    with pytest.raises(ValueError, match="same number of dimensions"):
        BlockLayout({"a": (3, 2), "b": (3, 2, 1)}, axis=0)
    with pytest.raises(ValueError, match="positive"):
        BlockLayout({"a": (3, 0)}, axis=0)
    with pytest.raises(ValueError, match="positive"):
        BlockLayout({"a": (3, 2.0)}, axis=None)
    with pytest.raises(ValueError, match="out of range"):
        BlockLayout({"a": (3, 2)}, axis=2)
    with pytest.raises(ValueError, match="dict"):
        BlockLayout([("a", (3, 2))], axis=0)
    with pytest.raises(TypeError):
        BlockLayout({"a": (3, 2)})  # axis is required


def test_negative_axis_is_normalised():
    layout = BlockLayout({"a": (3, 2, 1), "b": (3, 2, 2)}, axis=-1)
    assert layout.axis == 2
    assert layout.shape == (3, 2, 3)


# -- 4. equivalence with hand-written place closures (S1 listings) ----------------


def _packed_bed_blocks(n_z, n_r, alpha=0.5, biot=(5.0, 10.0), delta=(1.0, 0.5)):
    m = 2
    bi, de = np.reshape(biot, (1, 1, m)), np.reshape(delta, (1, 1, m))
    shape_p, shape_b = (n_z, n_r, m), (n_z, 1, m)
    z_f, r_f = np.linspace(0.0, 1.0, n_z + 1), np.linspace(0.0, 1.0, n_r + 1)
    bc_b = ({"a": 0, "b": 1, "d": [[1.0, 0.0]]}, {"a": 1, "b": 0, "d": 0})
    conv, conv_bc = construct_convflux_upwind(shape_b, z_f, bc=bc_b, v=1.0, axis=0)
    div_z = construct_div(shape_b, z_f, nu=0, axis=0)
    bc_p = ({"a": 1, "b": 0, "d": 0}, {"a": 1, "b": bi / de, "d": bi / de})
    grad, _, grad_b = construct_grad(shape_p, r_f, bc=bc_p, axis=1, shapes_d=(None, shape_b))
    div_r = construct_div(shape_p, r_f, nu=2, axis=1)
    shape_f = (n_z, n_r + 1, m)
    weights = np.broadcast_to(de, shape_f).reshape(-1, 1)
    flux, flux_b = -grad.multiply(weights), -grad_b.multiply(weights)
    outer = update_array_indices(eye_array(n_z * m), (shape_b, shape_b), (shape_b, shape_f),
                                 offset=(None, (0, n_r, 0)))
    blocks = [(div_r @ flux, "particle", "particle"),
              (div_r @ flux_b, "particle", "bulk"),
              (div_z @ conv - 3 * alpha * outer @ flux_b, "bulk", "bulk"),
              (-3 * alpha * outer @ flux, "bulk", "particle")]
    return shape_p, shape_b, blocks, (div_z @ conv_bc).toarray()


def test_packed_bed_equivalence():
    n_z, n_r, m = 5, 4, 2
    shape_p, shape_b, blocks, bc_rows = _packed_bed_blocks(n_z, n_r)
    shape = (n_z, n_r + 1, m)
    at = {"particle": (0, 0, 0), "bulk": (0, n_r, 0)}
    shapes = {"particle": shape_p, "bulk": shape_b}

    def place(block, rows, cols, row_at, col_at):  # the listing's closure
        return update_array_indices(block, (rows, cols), (shape, shape), offset=(row_at, col_at))

    lin_ref = sum(place(b, shapes[r], shapes[c], at[r], at[c]) for b, r, c in blocks).tocsc()
    g_ref = np.zeros(shape)
    g_ref[:, n_r:] = bc_rows.reshape(shape_b)

    layout = BlockLayout({"particle": shape_p, "bulk": shape_b}, axis=1)
    assert layout.shape == shape
    lin = layout.assemble(blocks)
    g_0 = np.zeros(layout.size)
    layout.put(g_0, "bulk", bc_rows)
    assert isinstance(lin, csc_array)
    np.testing.assert_array_equal(lin.toarray(), lin_ref.toarray())
    np.testing.assert_array_equal(g_0, g_ref.ravel())


def test_channel_equivalence():
    n_x, n_y, D = 4, 3, 0.1
    shape_f, shape_w = (n_x, n_y, 1), (n_x, 1, 1)
    shape = (n_x, n_y + 1, 1)
    at = {"wall": (0, 0, 0), "fluid": (0, 1, 0)}
    shapes = {"wall": shape_w, "fluid": shape_f}
    x_f, x_c = generate_grid(n_x, [0, 4.0], generate_x_c=True)
    y_f, y_c = generate_grid(n_y, [0, 1.0], generate_x_c=True)
    u = 2 * y_c - y_c ** 2
    v = np.broadcast_to(u[None, :, None], (n_x + 1, n_y, 1))
    bc_x = ({"a": 0, "b": 1, "d": 1}, {"outflow": True})
    conv, conv_bc = construct_convflux_upwind(shape_f, x_f, x_c, bc=bc_x, v=v, axis=0)
    bc_wall = {"a": 1, "b": 0, "d": -1 / D}
    grad, grad_r, _ = construct_grad(shape_f, y_f, y_c, bc=(bc_wall, {"a": 1, "b": 0, "d": 0}),
                                     axis=1, shapes_d=(shape_w, None))
    div_x = construct_div(shape_f, x_f, nu=0, axis=0)
    div_y = construct_div(shape_f, y_f, nu=0, axis=1)
    c_s, c_s_r = construct_boundary_value_matrices(
        shape_f, y_f, y_c, bc=bc_wall, axis=1, bound_id=0, shape_d=shape_w)
    lin_blocks = [(div_x @ conv - D * (div_y @ grad), "fluid", "fluid"),
                  (eye_array(n_x), "wall", "wall"),
                  (-c_s, "wall", "fluid")]
    rate_blocks = [(-D * (div_y @ grad_r), "fluid", "wall"), (-c_s_r, "wall", "wall")]

    def place(block, rows, cols, row_at, col_at):  # the listing's closure
        return update_array_indices(block, (rows, cols), (shape, shape), offset=(row_at, col_at))

    def by_closure(blocks):
        return sum(place(b, shapes[r], shapes[c], at[r], at[c]) for b, r, c in blocks).tocsc()

    g_ref = np.zeros(shape)
    g_ref[:, 1:] = (div_x @ conv_bc).toarray().reshape(shape_f)

    layout = BlockLayout({"wall": shape_w, "fluid": shape_f}, axis=1)
    assert layout.shape == shape
    for blocks in (lin_blocks, rate_blocks):
        np.testing.assert_array_equal(layout.assemble(blocks).toarray(), by_closure(blocks).toarray())
    g_0 = np.zeros(layout.shape)
    layout.put(g_0, "fluid", (div_x @ conv_bc).toarray())
    np.testing.assert_array_equal(g_0, g_ref)


# -- 5. region-major ------------------------------------------------------------------


def test_region_major_index_and_pick():
    n_z = 3
    regions = {"anode": (n_z, 2, 3), "membrane": (n_z, 4, 1), "cathode": (n_z, 1, 2)}
    layout = BlockLayout(regions, axis=None)
    assert layout.shape == (n_z * (6 + 4 + 2),)
    start = 0
    for name, shape in regions.items():
        n = int(np.prod(shape))
        expected = start + np.arange(n).reshape(shape)  # (z, x, field) order within the region
        np.testing.assert_array_equal(layout.index(name), expected)
        dense = np.zeros((n, layout.size))
        dense[np.arange(n), expected.ravel()] = 1.0
        np.testing.assert_array_equal(layout.pick(name).toarray(), dense)
        start += n
    # one field of one region, as pcec's RegionLayout.global_index(name, (field,))
    np.testing.assert_array_equal(layout.index("anode")[:, :, 1],
                                  np.arange(18).reshape(n_z, 2, 3)[:, :, 1])


# -- 6. fields as domains (B3.2 pattern) ----------------------------------------------


def test_fields_as_domains():
    n = 6
    x_f = np.linspace(0.0, 1.0, n + 1)
    bc = ({"a": 0, "b": 1, "d": 1.0}, {"a": 1, "b": 0, "d": 0.0})
    grad, grad_bc = construct_grad((n, 1), x_f, bc=bc, axis=0)
    div = construct_div((n, 1), x_f, nu=0, axis=0)
    lap, bvec = (div @ grad).tocoo(), (div @ grad_bc).toarray().ravel()

    layout = BlockLayout({"psi": (n, 1), "xi": (n, 1)}, axis=-1)
    assert layout.shape == (n, 2)
    hand = coo_array((lap.data, (2 * lap.row, 2 * lap.col)), shape=(2 * n, 2 * n))
    np.testing.assert_array_equal(layout.place(lap, "psi", "psi").toarray(), hand.toarray())
    hand_xi = coo_array((lap.data, (2 * lap.row + 1, 2 * lap.col)), shape=(2 * n, 2 * n))
    np.testing.assert_array_equal(layout.place(lap, "xi", "psi").toarray(), hand_xi.toarray())

    b = np.zeros(layout.shape)
    layout.put(b, "psi", bvec)
    b_hand = np.zeros(2 * n)
    b_hand[0::2] = bvec
    np.testing.assert_array_equal(b.ravel(), b_hand)


# -- 7. assemble ----------------------------------------------------------------------


def test_assemble_empty_and_duplicates():
    layout = BlockLayout({"a": (3, 1), "b": (2, 1)}, axis=0)
    empty = layout.assemble([])
    assert isinstance(empty, csc_array)
    assert empty.shape == (5, 5) and empty.nnz == 0

    block = _random_block(3, 2, "csr")
    full = _random_block(5, 5, "coo")
    total = layout.assemble([(block, "a", "b"), (block, "a", "b"), (full, None, None)])
    assert isinstance(total, csc_array)
    expected = 2 * layout.place(block, "a", "b").toarray() + full.toarray()
    np.testing.assert_allclose(total.toarray(), expected, rtol=0, atol=1e-15)

    with pytest.raises(ValueError, match="expected"):
        layout.assemble([(_random_block(3, 4, "csr"), "a", None)])  # does not span the state


# -- 8. per-cell field exchange with construct_coefficient_matrix ------------------------


def test_per_cell_exchange_idiom():
    """kappa (x_0 - x_1) in row 0 and kappa (x_1 - x_0) in row 1 of every cell of (n, 2)."""
    n = 5
    kappa = RNG.random((n, 1, 1)) + 0.5
    exchange = construct_coefficient_matrix(kappa * np.array([[1.0, -1.0], [-1.0, 1.0]]),
                                            shape=((n, 2, 1), (n, 1, 2)))
    k = kappa.ravel()
    i = np.arange(n)
    rows = np.concatenate([2 * i, 2 * i, 2 * i + 1, 2 * i + 1])
    cols = np.concatenate([2 * i, 2 * i + 1, 2 * i, 2 * i + 1])
    vals = np.concatenate([k, -k, -k, k])
    hand = coo_array((vals, (rows, cols)), shape=(2 * n, 2 * n))
    assert exchange.shape == (2 * n, 2 * n)
    np.testing.assert_array_equal(exchange.toarray(), hand.toarray())
