"""Deterministic checks for pymrm models.

Small tools for the verification steps every model needs, so that they do not
have to be rewritten (and debugged) for each model:

* :func:`check_jacobian` compares a Jacobian with finite differences;
* :func:`observed_orders` runs a refinement study and reports observed orders;
* :func:`find_roots` locates every sign change of a scalar function in a range;
* :func:`residual_check` judges a converged solution by a scale-free backward
  error instead of an absolute residual threshold.

The module depends on numpy and scipy only.
"""

import math

import numpy as np
from scipy import sparse
from scipy.optimize import brentq


def _as_dense_vector(g):
    return np.asarray(g.toarray() if sparse.issparse(g) else g, dtype=float).ravel()


def check_jacobian(fun, x, n_probe=4, rtol=1e-4, seed=0, eps=None):
    """Compare the Jacobian returned by ``fun`` with central differences.

    Parameters
    ----------
    fun : callable
        ``fun(x) -> (g, jac)``, the residual and its Jacobian, as passed to
        :func:`pymrm.newton`. ``jac`` must be a matrix (not a factorisation).
    x : numpy.ndarray
        State at which to compare. Use a state away from the solution, where
        all terms are active.
    n_probe : int, optional
        Number of random directions ``v`` for which ``jac @ v`` is compared with
        a central difference of ``g``.
    rtol : float, optional
        Tolerance on the relative mismatch.
    seed : int, optional
        Seed of the random directions (the check is deterministic).
    eps : float, optional
        Finite-difference step relative to the size of ``x``; default
        ``cbrt(machine epsilon)``.

    Returns
    -------
    dict
        ``ok`` (bool), ``max_rel_error`` over the probes, and, for at most 500
        unknowns, ``worst_entry`` ``(row, col, jac_value, fd_value)`` from a full
        column-by-column comparison.
    """
    x = np.asarray(x, dtype=float)
    shape = x.shape
    x_flat = x.ravel()
    g0, jac = fun(x.copy())
    g0 = _as_dense_vector(g0)
    jac = sparse.csr_array(jac) if sparse.issparse(jac) else np.asarray(jac, dtype=float)
    scale = max(np.max(np.abs(x_flat)), 1.0)
    h = (eps if eps is not None else np.finfo(float).eps ** (1.0 / 3.0)) * scale

    def g_at(xf):
        return _as_dense_vector(fun(xf.reshape(shape).copy())[0])

    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_probe):
        v = rng.standard_normal(x_flat.size)
        fd = (g_at(x_flat + h * v) - g_at(x_flat - h * v)) / (2.0 * h)
        jv = jac @ v
        denominator = max(np.max(np.abs(fd)), np.max(np.abs(jv)), np.max(np.abs(g0)), 1e-300)
        worst = max(worst, np.max(np.abs(jv - fd)) / denominator)
    result = {"ok": bool(worst <= rtol), "max_rel_error": float(worst), "worst_entry": None}

    if x_flat.size <= 500:
        dense = jac.toarray() if sparse.issparse(jac) else jac
        fd_full = np.empty_like(dense)
        for j in range(x_flat.size):
            e = np.zeros(x_flat.size)
            e[j] = h
            fd_full[:, j] = (g_at(x_flat + e) - g_at(x_flat - e)) / (2.0 * h)
        diff = np.abs(dense - fd_full)
        row, col = np.unravel_index(np.argmax(diff), diff.shape)
        result["worst_entry"] = (int(row), int(col), float(dense[row, col]), float(fd_full[row, col]))
    return result


def observed_orders(solve, ns, ratio=None):
    """Run a refinement study and report observed orders of convergence.

    Parameters
    ----------
    solve : callable
        ``solve(n) -> float or array``, the quantity of interest computed with
        resolution ``n`` (cells, or number of time steps).
    ns : sequence[int]
        At least three resolutions with a constant refinement ratio, for example
        ``(50, 100, 200, 400)``.
    ratio : float, optional
        Refinement ratio; default ``ns[1] / ns[0]``.

    Returns
    -------
    dict
        ``values`` per resolution, ``orders`` from each consecutive triple,
        ``extrapolated`` (Richardson estimate from the finest triple) and
        ``error_estimate`` of the finest value against it.

    Notes
    -----
    The order from values ``q1, q2, q3`` on successively finer grids is
    ``log(|q1 - q2| / |q2 - q3|) / log(ratio)``; it needs no exact solution. An
    order far from the expected one means the study is not in the asymptotic
    range, the refinement does not control the error that dominates, or there is
    a bug. For a non-uniform grid, refine it in a nested way (see the pymrm
    agent plugin's profiles.md).
    """
    ns = list(ns)
    if len(ns) < 3:
        raise ValueError("observed_orders needs at least three resolutions")
    ratio = float(ratio if ratio is not None else ns[1] / ns[0])
    values = [np.asarray(solve(n), dtype=float) for n in ns]
    orders = []
    for q1, q2, q3 in zip(values, values[1:], values[2:]):
        d1, d2 = np.max(np.abs(q1 - q2)), np.max(np.abs(q2 - q3))
        orders.append(float(math.log(d1 / d2) / math.log(ratio)) if d1 > 0 and d2 > 0 else float("nan"))
    p = orders[-1]
    q_prev, q_last = values[-2], values[-1]
    if np.isfinite(p) and p > 0:
        extrapolated = q_last + (q_last - q_prev) / (ratio**p - 1.0)
    else:
        extrapolated = q_last
    error_estimate = float(np.max(np.abs(q_last - extrapolated)))
    as_out = (lambda v: float(v)) if values[0].ndim == 0 else (lambda v: v)
    return {"values": [as_out(v) for v in values], "orders": orders,
            "extrapolated": as_out(extrapolated), "error_estimate": error_estimate}


def find_roots(f, lo, hi, n_scan=64, log=False, xtol=1e-12):
    """Locate every sign change of ``f`` on ``[lo, hi]`` and refine each root.

    Parameters
    ----------
    f : callable
        Scalar function. It may raise, or return ``nan``, where the underlying
        model cannot be evaluated; such points are reported, never taken as a
        sign.
    lo, hi : float
        Physically possible range.
    n_scan : int, optional
        Number of scan points.
    log : bool, optional
        Scan on a logarithmic grid (``lo`` and ``hi`` must be positive).
    xtol : float, optional
        Absolute tolerance of each refined root (``brentq`` ``xtol``).

    Returns
    -------
    dict
        ``roots`` (sorted list), ``failed`` (scan points where ``f`` could not
        be evaluated) and ``message``. No sign change gives an empty list and
        says so; it does NOT widen the range.
    """
    grid = np.geomspace(lo, hi, n_scan) if log else np.linspace(lo, hi, n_scan)

    def safe(x):
        try:
            value = float(f(x))
        except Exception:  # noqa: BLE001 - a failed model evaluation is data here
            return float("nan")
        return value

    values = np.array([safe(x) for x in grid])
    failed = [float(x) for x, v in zip(grid, values) if not np.isfinite(v)]
    roots = [float(x) for x, v in zip(grid, values) if v == 0.0]
    for (x1, v1), (x2, v2) in zip(zip(grid, values), zip(grid[1:], values[1:])):
        if np.isfinite(v1) and np.isfinite(v2) and v1 * v2 < 0.0:
            roots.append(float(brentq(safe, x1, x2, xtol=xtol)))
    roots.sort()
    if not roots:
        message = "no sign change in the scanned range"
    elif len(roots) > 1:
        message = f"{len(roots)} roots: multiplicity, report all of them"
    else:
        message = "one root"
    if failed:
        message += f"; f could not be evaluated at {len(failed)} scan points"
    return {"roots": roots, "failed": failed, "message": message}


def residual_check(fun, x, tol=1e-8):
    """Judge a solution by its componentwise backward error.

    An absolute threshold on the residual depends on the units and scaling of
    the equations. The backward error ``max_i |g_i| / (|J| |x| + |J x - g|)_i``
    is scale-free: it measures the relative change in the (linearised) equation
    coefficients that would make ``x`` exact.

    Parameters
    ----------
    fun : callable
        ``fun(x) -> (g, jac)``.
    x : numpy.ndarray
        Candidate solution, for example ``result.x`` from :func:`pymrm.newton`.
    tol : float, optional
        Acceptance threshold on the backward error.

    Returns
    -------
    dict
        ``ok`` (bool), ``backward_error`` and ``max_abs_residual``.
    """
    x = np.asarray(x, dtype=float)
    g, jac = fun(x.copy())
    g = _as_dense_vector(g)
    x_flat = x.ravel()
    jac = sparse.csr_array(jac) if sparse.issparse(jac) else np.asarray(jac, dtype=float)
    abs_jac = abs(jac)
    denominator = abs_jac @ np.abs(x_flat) + np.abs(jac @ x_flat - g)
    denominator = np.where(denominator > 0.0, denominator, np.finfo(float).tiny)
    backward_error = float(np.max(np.abs(g) / denominator))
    return {"ok": bool(backward_error <= tol), "backward_error": backward_error,
            "max_abs_residual": float(np.max(np.abs(g)))}
