import pytest
import numpy as np
from scipy.sparse import issparse
from pymrm import NumJac
from pymrm.numjac import (
    expand_dependencies,
    generate_sparsity_pattern,
    stencil_block_diagonals,
    colgroup,
)


# ---------------------------------------------------------------------------
# Existing test
# ---------------------------------------------------------------------------

def test_numjac_basic():
    shape = (5,)
    numjac = NumJac(shape)

    def f(x):
        return x**2

    x = np.arange(5.0)
    g, jac = numjac(f, x)
    assert g.shape == (5,)
    assert jac.shape[0] == 5
    assert jac.shape[1] == 5


# ---------------------------------------------------------------------------
# expand_dependencies
# ---------------------------------------------------------------------------

def test_expand_dependencies_shorthand():
    """Shorthand notation (tuple of ints) should be normalised."""
    shape_in = (5,)
    shape_out = (5,)
    deps = [(0,)]
    expanded = expand_dependencies(shape_in, shape_out, deps)
    assert len(expanded) > 0
    # Each entry is a 4-tuple: (idx_in, idx_out, fixed_axes, periodic_axes)
    for entry in expanded:
        assert len(entry) == 4


def test_expand_dependencies_full_notation():
    """Full (idx_in, idx_out, fixed_axes, periodic_axes) notation."""
    shape_in = (4, 4)
    shape_out = (4, 4)
    deps = [((0, 0), (0, 0), [], [])]
    expanded = expand_dependencies(shape_in, shape_out, deps)
    assert len(expanded) > 0


def test_expand_dependencies_three_tuple():
    """Three-tuple notation (no periodic axes)."""
    shape_in = (5, 5)
    shape_out = (5, 5)
    deps = [((0, 0), (0, 0), [])]
    expanded = expand_dependencies(shape_in, shape_out, deps)
    assert len(expanded) > 0


def test_expand_dependencies_with_slices():
    """Slice notation should expand to individual indices."""
    shape_in = (4,)
    shape_out = (4,)
    deps = [(slice(0, 4),)]
    expanded = expand_dependencies(shape_in, shape_out, deps)
    assert len(expanded) == 4


def test_expand_dependencies_single_tuple_wraps():
    """A single tuple (not a list of tuples) should be handled."""
    shape_in = (3,)
    shape_out = (3,)
    deps = (0,)  # single dependency tuple, not wrapped in a list
    expanded = expand_dependencies(shape_in, shape_out, deps)
    assert len(expanded) > 0


def test_expand_dependencies_invalid_raises():
    """Non-tuple dependency should raise ValueError."""
    with pytest.raises((ValueError, AttributeError)):
        expand_dependencies((5,), (5,), [42])


# ---------------------------------------------------------------------------
# generate_sparsity_pattern
# ---------------------------------------------------------------------------

def test_generate_sparsity_pattern_diagonal():
    """Zero offset → main diagonal pattern."""
    shape_in = (5,)
    shape_out = (5,)
    deps = expand_dependencies(shape_in, shape_out, [(0,)])
    rows, cols = generate_sparsity_pattern(shape_in, shape_out, deps)
    assert len(rows) == len(cols)
    assert len(rows) > 0


def test_generate_sparsity_pattern_tridiagonal():
    """Offsets -1, 0, +1 → tridiagonal pattern."""
    shape_in = (6,)
    shape_out = (6,)
    deps = expand_dependencies(shape_in, shape_out, [(-1,), (0,), (1,)])
    rows, cols = generate_sparsity_pattern(shape_in, shape_out, deps)
    assert len(rows) == len(cols)
    assert len(rows) > 6  # more than diagonal


# ---------------------------------------------------------------------------
# stencil_block_diagonals
# ---------------------------------------------------------------------------

def test_stencil_block_diagonals_default():
    """Default stencil should return a list of dependencies."""
    deps = stencil_block_diagonals(ndims=1)
    assert isinstance(deps, list)
    assert len(deps) > 0


def test_stencil_block_diagonals_2d():
    """2D stencil with axes_diagonals specified."""
    deps = stencil_block_diagonals(ndims=2, axes_diagonals=[0], axes_blocks=[1])
    assert isinstance(deps, list)
    assert len(deps) >= 1


def test_stencil_block_diagonals_too_many_axes_raises():
    """More axes than dimensions should raise ValueError."""
    with pytest.raises(ValueError):
        stencil_block_diagonals(ndims=1, axes_diagonals=[0, 1])


# ---------------------------------------------------------------------------
# colgroup
# ---------------------------------------------------------------------------

def test_colgroup_from_sparse_array():
    """colgroup should group columns by non-overlapping rows."""
    from scipy.sparse import csc_array
    # Simple tridiagonal sparsity
    n = 6
    rows = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
    cols = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 5])
    S = csc_array((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    g, num_groups = colgroup(S)
    assert len(g) == n
    assert num_groups >= 1


def test_colgroup_from_arrays():
    """colgroup accepts row and col index arrays."""
    n = 4
    rows = np.array([0, 1, 2, 3])
    cols = np.array([0, 1, 2, 3])
    g, num_groups = colgroup(rows, cols, shape=(n, n))
    assert num_groups == 1  # diagonal: no overlapping rows per column


# ---------------------------------------------------------------------------
# NumJac class
# ---------------------------------------------------------------------------

def test_numjac_shape_in_shape_out():
    """NumJac with different shape_in and shape_out."""
    shape_in = (5,)
    shape_out = (5,)
    numjac = NumJac(shape_in=shape_in, shape_out=shape_out)

    def f(x):
        return x**3

    x = np.arange(1.0, 6.0)
    g, jac = numjac(f, x)
    assert g.shape == (5,)
    assert jac.shape == (5, 5)
    # Diagonal of Jacobian should be approx 3*x^2
    np.testing.assert_allclose(jac.diagonal(), 3 * x**2, rtol=1e-4)


def test_numjac_2d():
    """NumJac on a 2D array function."""
    shape = (4, 3)
    numjac = NumJac(shape)

    def f(x):
        return x**2

    x = np.arange(1.0, 13.0).reshape(shape)
    g, jac = numjac(f, x)
    assert g.shape == shape
    assert jac.shape == (12, 12)


def test_numjac_custom_stencil():
    """NumJac with a custom stencil list."""
    shape = (6,)
    stencil = stencil_block_diagonals(ndims=1, axes_diagonals=[0])
    numjac = NumJac(shape, stencil=stencil)

    def f(x):
        return x**2 + np.roll(x, 1)

    x = np.ones(6)
    g, jac = numjac(f, x)
    assert jac.shape == (6, 6)
    assert issparse(jac)


def test_numjac_with_precomputed_f_value():
    """NumJac should accept a pre-computed f_value to skip re-evaluation."""
    shape = (4,)
    numjac = NumJac(shape)

    def f(x):
        return x**2

    x = np.array([1.0, 2.0, 3.0, 4.0])
    f_value = f(x)
    g, jac = numjac(f, x, f_value=f_value)
    assert g is f_value
    assert jac.shape == (4, 4)


def test_numjac_both_shape_and_shape_in_raises():
    """Specifying both shape and shape_in/shape_out should raise ValueError."""
    with pytest.raises(ValueError):
        NumJac(shape=(5,), shape_in=(5,))


def test_numjac_missing_shape_raises():
    """Specifying neither shape nor shape_in/shape_out should raise ValueError."""
    with pytest.raises(ValueError):
        NumJac()


def test_numjac_eps_jac():
    """Custom eps_jac should be stored on the instance."""
    nj = NumJac(shape=(3,), eps_jac=1e-5)
    assert nj.eps_jac == pytest.approx(1e-5)


# ---------------------------------------------------------------------------
# expand_dependencies – additional path coverage
# ---------------------------------------------------------------------------

def test_expand_dependencies_with_list_of_slices():
    """List containing slices should be expanded via slice_to_list."""
    shape_in = (6,)
    shape_out = (6,)
    # index using a list that contains a slice
    deps = [((slice(0, 3),), (0,), [])]
    expanded = expand_dependencies(shape_in, shape_out, deps)
    # Should expand to 3 entries (for i=0,1,2)
    assert len(expanded) >= 3


def test_expand_dependencies_with_range_in_axis():
    """Range objects in axis specification should be expanded."""
    shape_in = (5,)
    shape_out = (5,)
    deps = [((range(0, 3),), (0,), [])]
    expanded = expand_dependencies(shape_in, shape_out, deps)
    assert len(expanded) >= 3


def test_expand_dependencies_with_list_containing_range():
    """List containing a range object should be expanded."""
    shape_in = (5,)
    shape_out = (5,)
    deps = [(([range(0, 2), 4],), (0,), [])]
    expanded = expand_dependencies(shape_in, shape_out, deps)
    # Should cover indices 0, 1, 4
    assert len(expanded) >= 3


def test_expand_dependencies_periodic_axes():
    """Periodic axes should be handled."""
    shape_in = (6, 6)
    shape_out = (6, 6)
    deps = [((0, 0), (0, 0), [], [0])]
    expanded = expand_dependencies(shape_in, shape_out, deps)
    assert len(expanded) > 0


def test_expand_dependencies_periodic_axes_same_size_check():
    """Periodic axis must have the same size in input and output."""
    shape_in = (6, 5)
    shape_out = (6, 4)  # different size in axis 1
    with pytest.raises(ValueError):
        expand_dependencies(shape_in, shape_out, [((0, 0), (0, 0), [], [1])])


def test_expand_dependencies_periodic_fixed_conflict():
    """An axis cannot be both fixed and periodic."""
    shape_in = (5, 5)
    shape_out = (5, 5)
    with pytest.raises(ValueError):
        expand_dependencies(shape_in, shape_out, [((0, 0), (0, 0), [0], [0])])


def test_expand_dependencies_fixed_axes_not_list_raises():
    """fixed_axes must be a list (or None)."""
    shape_in = (5,)
    shape_out = (5,)
    with pytest.raises(ValueError):
        expand_dependencies(shape_in, shape_out, [((0,), (0,), 0)])


def test_expand_dependencies_non_tuple_entry_raises():
    """Non-tuple dependency entry should raise ValueError."""
    shape_in = (5,)
    shape_out = (5,)
    with pytest.raises(ValueError):
        expand_dependencies(shape_in, shape_out, [12345])


# ---------------------------------------------------------------------------
# colgroup with non-square sparse array
# ---------------------------------------------------------------------------

def test_colgroup_non_square_no_reorder():
    """Non-square sparse matrix should set try_reorder=False."""
    rows = np.array([0, 1, 2])
    cols = np.array([0, 1, 2])
    shape = (4, 5)  # non-square
    g, num_groups = colgroup(rows, cols, shape=shape)
    assert len(g) == 5
    assert num_groups >= 1


def test_colgroup_invalid_input_raises():
    """colgroup with invalid input should raise ValueError."""
    with pytest.raises(ValueError):
        colgroup("invalid_input")


# ---------------------------------------------------------------------------
# NumJac - stencil=None raises ValueError
# ---------------------------------------------------------------------------

def test_numjac_stencil_none_raises():
    """Passing stencil=None explicitly should raise ValueError."""
    with pytest.raises(ValueError):
        NumJac(shape=(5,), stencil=None)


# ---------------------------------------------------------------------------
# NumJac with 2D stencil from stencil_block_diagonals
# ---------------------------------------------------------------------------

def test_numjac_2d_with_stencil_block_diagonals():
    """2D NumJac using stencil_block_diagonals."""
    shape = (4, 3)
    numjac = NumJac(shape, axes_diagonals=[0, 1], axes_blocks=[-1])

    def f(x):
        return x**2

    x = np.arange(1.0, 13.0).reshape(shape)
    g, jac = numjac(f, x)
    assert g.shape == shape
    np.testing.assert_allclose(jac.diagonal(), (2 * x).ravel(), rtol=1e-4)


# ---------------------------------------------------------------------------
# expand_axis edge cases
# ---------------------------------------------------------------------------

def test_expand_axis_list_with_slice():
    """List containing a slice should expand correctly via expand_dependencies."""
    shape_in = (8,)
    shape_out = (8,)
    # A list containing a slice – hits the isinstance(v, slice) branch in expand_axis
    deps = [(([slice(0, 3)],), (0,), [])]
    expanded = expand_dependencies(shape_in, shape_out, deps)
    assert len(expanded) == 3


def test_expand_axis_unsupported_element_in_list_raises():
    """An unsupported type inside a list should raise ValueError."""
    with pytest.raises(ValueError):
        expand_dependencies((5,), (5,), [(([3.14],), (0,), [])])


def test_expand_axis_unsupported_type_raises():
    """An unsupported axis type (e.g. a set) should raise ValueError."""
    with pytest.raises(ValueError):
        expand_dependencies((5,), (5,), [(({"a"},), (0,), [])])


# ---------------------------------------------------------------------------
# fixed_axes and periodic_axes normalization
# ---------------------------------------------------------------------------

def test_expand_dependencies_fixed_axes_none_in_3tuple():
    """In 3-tuple form, fixed_axes=None should be normalised to an empty list."""
    shape_in = (4,)
    shape_out = (4,)
    # 3-tuple form: (idx_in, idx_out, fixed_axes)
    deps = [((0,), (0,), None)]
    expanded = expand_dependencies(shape_in, shape_out, deps)
    assert len(expanded) > 0


def test_expand_dependencies_fixed_axes_not_list_in_3tuple_raises():
    """In 3-tuple form, a non-list fixed_axes should raise ValueError."""
    with pytest.raises(ValueError, match="fixed_axes_list must be a list or None"):
        expand_dependencies((5,), (5,), [((0,), (0,), 42)])


# ---------------------------------------------------------------------------
# generate_sparsity_pattern – shape_in != shape_out
# ---------------------------------------------------------------------------

def test_generate_sparsity_pattern_non_square():
    """Non-square (shape_in != shape_out) should work for valid dependencies."""
    shape_in = (6,)
    shape_out = (4,)
    deps = expand_dependencies(shape_in, shape_out, [(0,)])
    rows, cols = generate_sparsity_pattern(shape_in, shape_out, deps)
    assert np.all(rows < 4)
    assert np.all(cols < 6)


# ---------------------------------------------------------------------------
# expand_index with non-tuple idx_out triggers line 98
# ---------------------------------------------------------------------------

def test_expand_index_non_tuple_out_raises():
    """Non-tuple idx_out in 3-tuple form should raise ValueError."""
    with pytest.raises(ValueError, match="Index must be a tuple or None"):
        # 3-tuple form with idx_out as non-tuple (integer)
        expand_dependencies((5,), (5,), [((0,), 42, [])])


# ---------------------------------------------------------------------------
# idx_out == None in the expansion loop
# ---------------------------------------------------------------------------

def test_expand_dependencies_idx_out_none_no_fixed():
    """3-tuple form with idx_out=None and no fixed_axes should work."""
    shape_in = (5,)
    shape_out = (5,)
    # 3-tuple: (idx_in, idx_out, fixed_axes) where idx_out=None
    deps = [((0,), None, [])]
    expanded = expand_dependencies(shape_in, shape_out, deps)
    assert len(expanded) > 0


def test_expand_dependencies_idx_out_none_with_fixed_raises():
    """3-tuple form with idx_out=None and non-empty fixed_axes should raise."""
    shape_in = (5,)
    shape_out = (5,)
    with pytest.raises(ValueError, match="Fixed axes are not allowed"):
        expand_dependencies(shape_in, shape_out, [((0,), None, [0])])


# ---------------------------------------------------------------------------
# NumJac init_stencil with stencil=None raises
# ---------------------------------------------------------------------------

def test_numjac_init_stencil_none_in_call():
    """NumJac.init_stencil called with None should raise."""
    nj = NumJac(shape=(3,))  # created successfully
    with pytest.raises(ValueError, match="stencil"):
        nj.init_stencil(None)


# ---------------------------------------------------------------------------
# CSR / CSC equivalence for various sparsity patterns
# ---------------------------------------------------------------------------

def _assert_jac_csr_csc_equal(shape, f, x, atol=1e-8, **kwargs):
    """Helper: verify CSR and CSC NumJac produce identical Jacobians."""
    nj_csc = NumJac(shape=shape, format="csc", **kwargs)
    nj_csr = NumJac(shape=shape, format="csr", **kwargs)
    _, jac_csc = nj_csc(f, x)
    _, jac_csr = nj_csr(f, x)
    np.testing.assert_allclose(
        jac_csr.toarray(), jac_csc.toarray(), atol=atol,
        err_msg="CSR and CSC Jacobians differ",
    )
    return jac_csr, jac_csc


def test_numjac_csr_csc_block_diagonal_1d():
    """CSR and CSC Jacobians agree for a 1D block-diagonal stencil."""
    shape = (20,)
    x = np.linspace(1.0, 5.0, 20)
    _assert_jac_csr_csc_equal(shape, lambda c: c ** 2, x)


def test_numjac_csr_csc_tridiagonal():
    """CSR and CSC Jacobians agree for a tridiagonal stencil."""
    shape = (15,)
    stencil = stencil_block_diagonals(ndims=1, axes_diagonals=[0])
    x = np.linspace(1.0, 5.0, 15)

    def f(c):
        result = c.copy()
        result[1:] += 0.5 * c[:-1]
        result[:-1] += 0.3 * c[1:]
        return result

    _assert_jac_csr_csc_equal(shape, f, x, stencil=stencil)


def test_numjac_csr_csc_2d_block():
    """CSR and CSC Jacobians agree for a 2D block-diagonal stencil."""
    shape = (6, 4)
    x = np.arange(1.0, 25.0).reshape(shape)
    _assert_jac_csr_csc_equal(
        shape, lambda c: c ** 2, x,
        axes_diagonals=[0], axes_blocks=[1],
    )


def test_numjac_csr_csc_large_block_diagonal():
    """CSR and CSC Jacobians agree on a larger block-diagonal system."""
    N, B = 20, 10
    shape = (N, B)
    x = np.random.default_rng(42).standard_normal(shape) + 2.0

    def f(c):
        return c ** 3 - 2.0 * c

    jac_csr, jac_csc = _assert_jac_csr_csc_equal(
        shape, f, x, axes_diagonals=[], axes_blocks=[1],
    )
    # Diagonal should approximate 3*x^2 - 2
    np.testing.assert_allclose(
        jac_csr.diagonal(), (3.0 * x ** 2 - 2.0).ravel(), rtol=1e-3,
    )


def test_numjac_csr_csc_non_square():
    """CSR and CSC Jacobians agree when shape_in != shape_out."""
    shape_in = (6,)
    shape_out = (4,)
    x = np.arange(1.0, 7.0)

    nj_csc = NumJac(shape_in=shape_in, shape_out=shape_out, format="csc")
    nj_csr = NumJac(shape_in=shape_in, shape_out=shape_out, format="csr")

    def f(c):
        return c[:4] ** 2

    _, jac_csc = nj_csc(f, x)
    _, jac_csr = nj_csr(f, x)
    np.testing.assert_allclose(
        jac_csr.toarray(), jac_csc.toarray(), atol=1e-8,
    )


def test_numjac_csr_csc_with_precomputed_f_value():
    """CSR and CSC agree when f_value is precomputed."""
    shape = (8,)
    x = np.linspace(0.5, 4.0, 8)

    def f(c):
        return c ** 2

    f_val = f(x)
    nj_csr = NumJac(shape=shape, format="csr")
    nj_csc = NumJac(shape=shape, format="csc")
    _, jac_csr = nj_csr(f, x, f_value=f_val)
    _, jac_csc = nj_csc(f, x, f_value=f_val)
    np.testing.assert_allclose(
        jac_csr.toarray(), jac_csc.toarray(), atol=1e-8,
    )


def test_numjac_csr_csc_periodic():
    """CSR and CSC agree for a periodic stencil."""
    shape = (10,)
    x = np.ones(10) * 2.0
    stencil = stencil_block_diagonals(
        ndims=1, axes_diagonals=[0], periodic_axes=[0],
    )

    def f(c):
        return c ** 2 + np.roll(c, 1) + np.roll(c, -1)

    _assert_jac_csr_csc_equal(shape, f, x, stencil=stencil)


def test_numjac_csr_returns_csr_type():
    """NumJac with format='csr' returns a csr_array."""
    from scipy.sparse import csr_array as csr_cls
    shape = (5,)
    nj = NumJac(shape=shape, format="csr")
    _, jac = nj(lambda c: c ** 2, np.arange(1.0, 6.0))
    assert isinstance(jac, csr_cls)


def test_numjac_csc_returns_csc_type():
    """NumJac with format='csc' returns a csc_array."""
    from scipy.sparse import csc_array as csc_cls
    shape = (5,)
    nj = NumJac(shape=shape, format="csc")
    _, jac = nj(lambda c: c ** 2, np.arange(1.0, 6.0))
    assert isinstance(jac, csc_cls)
