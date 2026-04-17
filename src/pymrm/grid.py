"""Grid-generation utilities for one-dimensional coordinates."""

import numpy as np


def non_uniform_grid(left_bound, right_bound, num_points, dx_inf, factor):
    """Generate a one-dimensional stretched face grid.

    Parameters
    ----------
    left_bound, right_bound : float
        Domain bounds.
    num_points : int
        Number of returned face coordinates, including both boundaries.
    dx_inf : float
        Asymptotic cell width used in the stretching expression.
    factor : float
        Geometric stretching factor. Values larger than ``1`` stretch cells;
        values between ``0`` and ``1`` compress cells.

    Returns
    -------
    numpy.ndarray
        Monotonic array of face coordinates with length ``num_points``.
    """
    a = np.log(factor)
    unif = np.arange(num_points)
    b = np.exp(-a * unif)
    length = right_bound - left_bound
    c = (np.exp(a * (length / dx_inf - num_points + 1.0)) - b[-1]) / (1 - b[-1])
    x_f = left_bound + unif * dx_inf + np.log((1 - c) * b + c) * (dx_inf / a)
    return x_f


def generate_grid(size, x_f=None, generate_x_c=False, x_c=None):
    """Return face coordinates and optionally cell-center coordinates.

    Parameters
    ----------
    size : int
        Number of cells along the axis.
    x_f : array_like, optional
        Face coordinates. Accepted inputs are:

        * ``None`` or an empty array: build a uniform grid on ``[0, 1]``;
        * an array of length ``size + 1``: interpreted directly as face
          coordinates;
        * an array-like of length ``2``: interpreted as ``(xmin, xmax)`` and
          used to build a uniform grid.
    generate_x_c : bool, optional
        If ``True``, also return cell-center coordinates.
    x_c : array_like, optional
        Explicit cell-center coordinates. When provided, length must equal
        ``size``.

    Returns
    -------
    numpy.ndarray or tuple[numpy.ndarray, numpy.ndarray]
        Face coordinates, and optionally cell-center coordinates.

    Raises
    ------
    ValueError
        If provided coordinates are inconsistent with ``size``.
    """
    if x_f is None or len(x_f) == 0:
        # Default to a uniform grid between 0 and 1 if x_f is not provided
        x_f = np.linspace(0.0, 1.0, size + 1)
    elif len(x_f) == size + 1:
        x_f = np.asarray(x_f)
    elif len(x_f) == 2:
        # Create uniform grid between specified boundaries
        x_f = np.linspace(x_f[0], x_f[1], size + 1)
    else:
        raise ValueError("Grid cannot be generated: check 'x_f' length.")

    if generate_x_c:
        if x_c is None:
            # Compute midpoints if no cell-centered grid is provided
            x_c = 0.5 * (x_f[1:] + x_f[:-1])
        elif len(x_c) == size:
            x_c = np.asarray(x_c)
        else:
            raise ValueError("Cell-centered grid not properly defined.")
        return x_f, x_c

    return x_f
