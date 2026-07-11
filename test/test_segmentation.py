"""Tests for SDF-based domain segmentation (:mod:`pymrm.segmentation`)."""

import numpy as np
import pytest
from scipy.sparse import coo_array

from pymrm.grid import generate_grid
from pymrm.ibm import construct_ibm, apply_ibm, _expand_full
from pymrm.ibm_recon import construct_ibm_normal_derivative
from pymrm.ibm_coupling import (
    construct_ibm_interface_values, apply_ibm_interface,
)
from pymrm.operators import construct_grad
from pymrm.segmentation import (
    segment_domain, crossing_segments, segment_values, wall_contact,
    wall_patch, wall_values, combine_interface_conditions, segment_field,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid_1d(n):
    x_f = np.linspace(0.0, 1.0, n + 1)
    _, x_c = generate_grid(n, x_f, generate_x_c=True)
    return x_c


def _two_slabs_1d(n=20):
    x_c = _grid_1d(n)
    sdf = np.minimum(np.abs(x_c - 0.275) - 0.075, np.abs(x_c - 0.70) - 0.10)
    return x_c, sdf


def _two_disks_2d(n=24, r=0.12, c1=(0.3, 0.3), c2=(0.72, 0.7)):
    x_c = _grid_1d(n)
    xx, yy = np.meshgrid(x_c, x_c, indexing="ij")
    sdf = np.minimum(np.hypot(xx - c1[0], yy - c1[1]) - r,
                     np.hypot(xx - c2[0], yy - c2[1]) - r)
    return x_c, sdf


def _ghost_extraction_full(ibm, value=1.0):
    fr = np.concatenate([_expand_full(ibm, ibm.row_out).ravel(),
                         _expand_full(ibm, ibm.row_in).ravel()])
    fg = np.concatenate([_expand_full(ibm, ibm.ghost_out).ravel(),
                         _expand_full(ibm, ibm.ghost_in).ravel()])
    data = np.full(fr.size, float(value))
    return coo_array((data, (fr, fg)),
                     shape=(ibm.n_cells, ibm.n_cells)).tocsr()


# ---------------------------------------------------------------------------
# segment_domain / crossing_segments
# ---------------------------------------------------------------------------

def test_segment_domain_1d():
    _, sdf = _two_slabs_1d(n=20)
    seg = segment_domain(sdf)
    assert seg.n_segments == 2
    assert seg.region == "negative"
    assert seg.sizes.sum() == int(np.sum(sdf < 0))
    assert np.array_equal(np.unique(seg.labels), [0, 1, 2])


def test_segment_domain_2d_disjoint():
    _, sdf = _two_disks_2d()
    seg = segment_domain(sdf)
    assert seg.n_segments == 2
    assert seg.sizes.sum() == int(np.sum(sdf < 0))


def test_connectivity_merges_diagonal():
    """Two solids touching only at a corner: separate for faces, one for diag."""
    grid = np.ones((5, 5))
    grid[1, 1] = -1.0
    grid[2, 2] = -1.0
    assert segment_domain(grid, connectivity=1).n_segments == 2
    assert segment_domain(grid, connectivity=2).n_segments == 1


def test_crossing_segments_per_disk_constant():
    x_c, sdf = _two_disks_2d()
    seg = segment_domain(sdf)
    ibm = construct_ibm(sdf, [x_c, x_c])
    cs = crossing_segments(seg, ibm)
    assert cs.shape == (ibm.n_crossings,)
    assert set(np.unique(cs).tolist()) == {1, 2}
    assert np.all(cs >= 1)


def test_crossing_segments_positive_side():
    x_c, sdf = _two_disks_2d()
    seg_pos = segment_domain(sdf, region="positive")
    ibm = construct_ibm(sdf, [x_c, x_c])
    cs = crossing_segments(seg_pos, ibm)
    # Outside is a single connected fluid region for two separate disks.
    assert seg_pos.n_segments == 1
    assert np.all(cs == 1)


def test_crossing_segments_shape_mismatch():
    x_c, sdf = _two_slabs_1d(n=20)
    seg = segment_domain(sdf)
    ibm = construct_ibm(sdf[:-1], _grid_1d(19))  # different spatial shape
    with pytest.raises(ValueError, match="spatial_shape"):
        crossing_segments(seg, ibm)


# ---------------------------------------------------------------------------
# wall_contact / isolated pocket detection
# ---------------------------------------------------------------------------

def test_wall_contact_disks_do_not_touch():
    _, sdf = _two_disks_2d()
    wc = wall_contact(segment_domain(sdf))
    assert wc.shape == (2, 2, 2)
    assert not wc.any()


def test_isolated_fluid_pocket():
    """A solid ring encloses a fluid pocket that touches no domain wall."""
    n = 40
    x_c = _grid_1d(n)
    xx, yy = np.meshgrid(x_c, x_c, indexing="ij")
    r = np.hypot(xx - 0.5, yy - 0.5)
    sdf = np.abs(r - 0.25) - 0.05          # solid ring 0.20 < r < 0.30
    seg = segment_domain(sdf, region="positive")
    contact = wall_contact(seg).any(axis=(1, 2))
    pockets = np.flatnonzero(~contact) + 1
    assert pockets.size == 1               # the enclosed inner disk
    inner = pockets[0]
    # The pocket is the region nearest the centre.
    assert seg.labels[n // 2, n // 2] == inner


# ---------------------------------------------------------------------------
# segment_values
# ---------------------------------------------------------------------------

def _interleaved_disks(n=24, npn=2, ncn=3):
    x_c, sdf = _two_disks_2d(n=n)
    ibm = construct_ibm(sdf, [x_c, x_c], axes=(0, 2),
                        shape=(n, npn, n, ncn), rescale=False)
    seg = segment_domain(sdf)
    return x_c, sdf, ibm, seg


def test_segment_values_maps_and_broadcasts():
    x_c, sdf, ibm, seg = _interleaved_disks()
    npn, ncn = 2, 3
    seg_ids = crossing_segments(seg, ibm)
    rng = np.random.default_rng(0)
    v1 = rng.random((npn, ncn))
    v2 = rng.random((npn, ncn))
    out = segment_values({1: v1, 2: v2}, seg, ibm)
    assert out.shape == (ibm.n_crossings, npn, ncn)
    table = np.stack([v1, v2])
    manual = table[seg_ids - 1]
    np.testing.assert_allclose(out, manual)
    # Accepted by apply_ibm as canonical point data.
    A = _ghost_extraction_full(ibm)
    _, g_seg = apply_ibm(A, ibm, values_outside=out)
    _, g_manual = apply_ibm(A, ibm, values_outside=manual)
    np.testing.assert_allclose(g_seg, g_manual)


def test_segment_values_scalar_per_segment_pads_ns():
    x_c, sdf, ibm, seg = _interleaved_disks()
    out = segment_values({1: 0.0, 2: 5.0}, seg, ibm)
    assert out.shape == (ibm.n_crossings, 1, 1)
    A = _ghost_extraction_full(ibm)
    apply_ibm(A, ibm, values_outside=out)   # must not raise


def test_segment_values_array_input():
    x_c, sdf = _two_disks_2d()
    seg = segment_domain(sdf)
    ibm = construct_ibm(sdf, [x_c, x_c])
    arr = np.array([2.0, 7.0])              # (n_segments,)
    out = segment_values(arr, seg, ibm)
    manual = arr[crossing_segments(seg, ibm) - 1]
    np.testing.assert_allclose(out, manual)


def test_segment_values_missing_label_errors():
    x_c, sdf = _two_disks_2d()
    seg = segment_domain(sdf)
    ibm = construct_ibm(sdf, [x_c, x_c])
    with pytest.raises(ValueError, match="no value for segment"):
        segment_values({1: 1.0}, seg, ibm)          # label 2 missing
    out = segment_values({1: 1.0}, seg, ibm, default=9.0)
    manual = np.where(crossing_segments(seg, ibm) == 1, 1.0, 9.0)
    np.testing.assert_allclose(out, manual)


# ---------------------------------------------------------------------------
# segment_field
# ---------------------------------------------------------------------------

def test_segment_field_per_cell():
    x_c, sdf = _two_disks_2d()
    seg = segment_domain(sdf)
    field = segment_field({1: 3.0, 2: 8.0}, seg, default=1.0)
    assert field.shape == seg.labels.shape
    assert np.all(field[seg.labels == 1] == 3.0)
    assert np.all(field[seg.labels == 2] == 8.0)
    assert np.all(field[seg.labels == 0] == 1.0)


# ---------------------------------------------------------------------------
# wall_patch / wall_values
# ---------------------------------------------------------------------------

def _wall_touching_disk(n=24, npn=2, ncn=3):
    """One disk crossing the lower-x wall, one interior disk."""
    x_c = _grid_1d(n)
    xx, yy = np.meshgrid(x_c, x_c, indexing="ij")
    sdf = np.minimum(np.hypot(xx - 0.0, yy - 0.5) - 0.18,
                     np.hypot(xx - 0.7, yy - 0.5) - 0.12)
    ibm = construct_ibm(sdf, [x_c, x_c], axes=(0, 2),
                        shape=(n, npn, n, ncn), rescale=False)
    seg = segment_domain(sdf)
    return x_c, sdf, ibm, seg


def test_wall_patch_shape_and_labels():
    n, npn, ncn = 24, 2, 3
    x_c, sdf, ibm, seg = _wall_touching_disk(n=n, npn=npn, ncn=ncn)
    patch = wall_patch(seg, ibm, axis=0, side="lower")
    # full-field rank with wall axis (0) and ns axes (1, 3) at size 1:
    assert patch.shape == (1, 1, n, 1)
    # Labels on the wall match seg.labels[0, :]
    np.testing.assert_array_equal(patch.ravel(), seg.labels[0, :])
    assert patch.max() >= 1                 # the wall-touching disk reaches it


def test_wall_patch_where_builds_coefficient():
    n, npn, ncn = 24, 2, 3
    x_c, sdf, ibm, seg = _wall_touching_disk(n=n, npn=npn, ncn=ncn)
    patch = wall_patch(seg, ibm, axis=0, side="lower")
    touching = int(seg.labels[0, seg.labels[0] > 0][0])
    d = np.where(patch == touching, 1.0, 0.0)
    # construct_grad accepts it as a boundary coefficient without error.
    x_f = np.linspace(0.0, 1.0, n + 1)
    bc = ({"a": 0.0, "b": 1.0, "d": d}, None)
    _, grad_bc = construct_grad((n, npn, n, ncn), x_f, x_c, bc=bc, axis=0)
    assert grad_bc.nnz > 0


def test_wall_values_matches_manual_where():
    n, npn, ncn = 24, 2, 3
    x_c, sdf, ibm, seg = _wall_touching_disk(n=n, npn=npn, ncn=ncn)
    rng = np.random.default_rng(2)
    v = {int(seg.labels[0, seg.labels[0] > 0][0]): rng.random((npn, ncn))}
    label = next(iter(v))
    wv = wall_values(v, seg, ibm, axis=0, side="lower", default=0.0)
    assert wv.shape == (1, npn, n, ncn)
    patch = wall_patch(seg, ibm, axis=0, side="lower")        # (1,1,n,1)
    manual = np.where(patch == label, v[label][None, :, None, :], 0.0)
    np.testing.assert_allclose(wv, manual)


# ---------------------------------------------------------------------------
# combine_interface_conditions
# ---------------------------------------------------------------------------

def _conjugate_ic(D_out, D_in, K=1.0):
    return ({"a": (D_out, D_in), "b": (0.0, 0.0), "d": 0.0},
            {"a": (0.0, 0.0), "b": (1.0, -K), "d": 0.0})


def test_combine_interface_conditions_equals_manual():
    x_c, sdf = _two_disks_2d(n=24)
    ibm = construct_ibm(sdf, [x_c, x_c])
    recon = construct_ibm_normal_derivative(ibm, sdf, [x_c, x_c])
    seg = segment_domain(sdf)
    seg_ids = crossing_segments(seg, ibm)

    ic_by_seg = {1: _conjugate_ic(1.0, 5.0, K=1.0),
                 2: _conjugate_ic(1.0, 0.5, K=2.5)}
    ic = combine_interface_conditions(ic_by_seg, seg, ibm)

    # Manual per-crossing ic: pick each coefficient by owning segment.
    D_in = np.where(seg_ids == 1, 5.0, 0.5)
    K = np.where(seg_ids == 1, 1.0, 2.5)
    ic_manual = ({"a": (np.ones(ibm.n_crossings), D_in),
                  "b": (0.0, 0.0), "d": 0.0},
                 {"a": (0.0, 0.0), "b": (np.ones(ibm.n_crossings), -K),
                  "d": 0.0})

    Ho1, ho1, Hi1, hi1 = construct_ibm_interface_values(ibm, recon, ic)
    Ho2, ho2, Hi2, hi2 = construct_ibm_interface_values(ibm, recon, ic_manual)
    assert np.abs((Ho1 - Ho2).toarray()).max() < 1e-12
    assert np.abs((Hi1 - Hi2).toarray()).max() < 1e-12
    np.testing.assert_allclose(ho1, ho2, atol=1e-12)
    np.testing.assert_allclose(hi1, hi2, atol=1e-12)


def test_combine_interface_conditions_default_and_missing():
    x_c, sdf = _two_disks_2d(n=24)
    ibm = construct_ibm(sdf, [x_c, x_c])
    seg = segment_domain(sdf)
    with pytest.raises(ValueError, match="no value for segment"):
        combine_interface_conditions({1: _conjugate_ic(1.0, 5.0)}, seg, ibm)
    ic = combine_interface_conditions(
        {1: _conjugate_ic(1.0, 5.0)}, seg, ibm,
        default=_conjugate_ic(1.0, 2.0))
    assert isinstance(ic, tuple) and len(ic) == 2
