import pytest
import numpy as np
from pymrm.grid import non_uniform_grid, generate_grid


def test_non_uniform_grid():
    grid = non_uniform_grid(0, 1, 10, 0.1, 0.75)
    assert np.all(np.diff(grid) > 0)
    assert grid[0] == 0
    assert np.isclose(grid[-1], 1)


def test_non_uniform_grid_length():
    n = 12
    grid = non_uniform_grid(0.0, 2.0, n, 0.05, 0.8)
    assert len(grid) == n
    assert grid[0] == pytest.approx(0.0)
    assert grid[-1] == pytest.approx(2.0)
    assert np.all(np.diff(grid) > 0)


def test_generate_grid():
    size = 10
    x_f = np.linspace(0, 1, size + 1)
    x_f_out, x_c_out = generate_grid(size, x_f, generate_x_c=True)
    assert len(x_f_out) == size + 1
    assert len(x_c_out) == size


def test_generate_grid_default_uniform():
    """Default behaviour (no x_f) should produce a uniform grid on [0, 1]."""
    size = 5
    x_f_out = generate_grid(size)
    assert len(x_f_out) == size + 1
    assert x_f_out[0] == pytest.approx(0.0)
    assert x_f_out[-1] == pytest.approx(1.0)
    assert np.allclose(np.diff(x_f_out), np.diff(x_f_out)[0])


def test_generate_grid_two_element_xf():
    """Two-element x_f specifies boundaries; uniform grid is constructed."""
    size = 8
    x_f_out = generate_grid(size, x_f=[0.0, 2.0])
    assert len(x_f_out) == size + 1
    assert x_f_out[0] == pytest.approx(0.0)
    assert x_f_out[-1] == pytest.approx(2.0)


def test_generate_grid_with_custom_xc():
    """User-provided x_c is passed through when generate_x_c=True."""
    size = 4
    x_f = np.linspace(0, 1, size + 1)
    x_c_custom = np.array([0.1, 0.3, 0.6, 0.9])
    x_f_out, x_c_out = generate_grid(size, x_f, generate_x_c=True, x_c=x_c_custom)
    np.testing.assert_array_equal(x_c_out, x_c_custom)


def test_generate_grid_wrong_xf_length_raises():
    """x_f with an invalid length should raise ValueError."""
    with pytest.raises(ValueError):
        generate_grid(5, x_f=np.linspace(0, 1, 4))


def test_generate_grid_wrong_xc_length_raises():
    """x_c with wrong length should raise ValueError."""
    with pytest.raises(ValueError):
        generate_grid(5, x_f=np.linspace(0, 1, 6), generate_x_c=True, x_c=np.array([0.1, 0.5]))


def test_generate_grid_cell_centers_are_midpoints():
    """Cell centers should be midpoints of the faces when x_c is not provided."""
    size = 6
    x_f = np.linspace(0, 3, size + 1)
    _, x_c = generate_grid(size, x_f, generate_x_c=True)
    expected_xc = 0.5 * (x_f[:-1] + x_f[1:])
    np.testing.assert_allclose(x_c, expected_xc)
