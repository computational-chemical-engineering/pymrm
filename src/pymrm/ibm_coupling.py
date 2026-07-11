"""General interface conditions at immersed boundaries for :mod:`pymrm`.

This module provides *Layer 2* of the generalized immersed-interface
coupling: it combines the one-sided normal-derivative reconstructions of
:mod:`pymrm.ibm_recon` with interface conditions specified in the same
dictionary format as :func:`pymrm.coupling.construct_interface_matrices`,
and eliminates the two unknown interface values per crossing.

Interface conditions
--------------------
``ic`` is a tuple of two equation dictionaries.  Each dictionary defines one
interface equation

.. math::
    a_\\text{out}\\, q_\\text{out} + a_\\text{in}\\, q_\\text{in}
    + b_\\text{out}\\, c_\\Gamma^\\text{out} + b_\\text{in}\\, c_\\Gamma^\\text{in}
    = d ,

with ``{"a": (a_out, a_in), "b": (b_out, b_in), "d": d}`` and the *outward
per side* derivative convention of :mod:`pymrm.ibm_recon` (``q_out`` out of
the fluid, ``q_in`` out of the solid).  With this convention the dictionaries
are numerically identical to a grid-aligned
:func:`~pymrm.coupling.construct_interface_matrices` call with the fluid as
subdomain 0.  Examples:

conjugate diffusion (flux + value continuity)::

    ic = ({"a": (D_out, D_in), "b": (0, 0), "d": 0},
          {"a": (0, 0), "b": (1, -1), "d": 0})

partition coefficient ``c_out = K c_in``::

    ic = ({"a": (D_out, D_in), "b": (0, 0), "d": 0},
          {"a": (0, 0), "b": (1, -K), "d": 0})

Coefficient broadcasting follows the canonical point shape
``(n_crossings, *ns_shape)`` with strict NumPy semantics (see
:func:`pymrm.ibm._normalize_point_values`): a scalar applies everywhere, a
``(nc,)`` array is per component, ``(n_crossings, 1, ..., 1)`` is per crossing,
and any array broadcastable to ``(n_crossings, *ns_shape)`` is accepted.  A
bare 1-D array of length ``n_crossings`` is per-crossing only for a purely
spatial field; when non-spatial axes are present reshape it to
``(n_crossings, 1, ..., 1)``.  The per-crossing/per-component 2x2 eliminations
are independent, so this linear path cannot couple components at the interface;
cross-component interface physics (Maxwell-Stefan, coupled surface reactions)
uses the augmented degree-of-freedom framework built on
:func:`pymrm.ibm_recon.construct_ibm_normal_derivative_ops`.

Usage
-----
The elimination expresses the interface values as sparse functions of the
field, ``c_gamma = H @ c + h``, which enter through the source matrices of
:func:`pymrm.apply_ibm`::

    ibm = construct_ibm(sdf, x_c)
    recon = construct_ibm_normal_derivative(ibm, sdf, x_c)
    A_final, g_final = apply_ibm_interface(A, ibm, recon, ic)
    # solve A_final @ c + g_final == rhs terms as usual (source is *added*)

Row conditioning from :func:`pymrm.construct_ibm` is handled automatically:
``H``/``h`` describe wall values in terms of the unscaled field and enter
only through the already-scaled ``G`` matrices.  Any independent right-hand
side assembled before the IBM must still pass through
:func:`pymrm.apply_ibm_vector`, exactly as for the Dirichlet IBM.
"""

import warnings
import numpy as np
from scipy.sparse import csr_array

from pymrm.ibm import apply_ibm, _normalize_point_values
from pymrm.ibm_recon import construct_ibm_normal_derivative_ops

__all__ = [
    "construct_ibm_interface_values",
    "apply_ibm_interface",
    "construct_ibm_boundary_values",
]


# ---------------------------------------------------------------------------
# Interface-condition coefficient handling
# ---------------------------------------------------------------------------

def _coerce_coeff(value, npnt, ns_shape, name):
    """Normalise one ic coefficient to an array of shape ``(npnt, ns_size)``.

    Delegates to :func:`pymrm.ibm._normalize_point_values`, so a coefficient
    follows the canonical point shape ``(npnt, *ns_shape)`` with strict NumPy
    broadcasting: scalar -> everywhere; ``(nc,)`` -> per component; ``(np, 1)``
    -> per phase; ``(npnt, 1, ..., 1)`` -> per crossing; etc.  A bare 1-D array
    of length ``npnt`` is per-crossing only for a purely spatial field.
    """
    return _normalize_point_values(value, npnt, ns_shape, name)


def _ic_coefficients(ic, npnt, ns_shape):
    """Normalise the two-equation ic tuple.

    Returns nested lists ``a[i][s]``, ``b[i][s]`` (equation ``i``, side ``s``;
    0 = out, 1 = in) and ``d[i]``, each of shape ``(npnt, ns_size)``.
    """
    if len(ic) != 2:
        raise ValueError(f"ic must contain exactly 2 equations, got {len(ic)}")
    zero = np.zeros((1, 1))
    a = [[zero, zero], [zero, zero]]
    b = [[zero, zero], [zero, zero]]
    d = [zero, zero]
    for i, eq in enumerate(ic):
        if not eq:
            continue
        for key, target in (("a", a), ("b", b)):
            if key in eq:
                pair = eq[key]
                if np.ndim(pair) == 0 or len(pair) != 2:
                    raise ValueError(
                        f"ic[{i}]['{key}'] must be a pair "
                        "(coefficient_out, coefficient_in)"
                    )
                for s in range(2):
                    target[i][s] = _coerce_coeff(
                        pair[s], npnt, ns_shape, f"ic[{i}]['{key}'][{s}]")
        if "d" in eq:
            d[i] = _coerce_coeff(eq["d"], npnt, ns_shape, f"ic[{i}]['d']")
    return a, b, d


def _scale_rows(mat, factors):
    """Return a copy of CSR *mat* with row ``r`` multiplied by ``factors[r]``."""
    out = csr_array(mat, copy=True)
    out.data = out.data * np.repeat(factors, np.diff(out.indptr))
    return out


def _guard_small(den, tol, what):
    """Zero out (and warn about) near-singular local denominators."""
    scale = np.max(np.abs(den)) if den.size else 0.0
    bad = np.abs(den) <= tol * max(scale, 1.0)
    if np.any(bad):
        crossings = np.unique(np.nonzero(bad)[0])
        warnings.warn(
            f"IBM interface: {what} nearly singular at "
            f"{crossings.size} crossing(s) (indices {crossings[:10]}"
            f"{'...' if crossings.size > 10 else ''}); their interface "
            "values are set to zero.",
            RuntimeWarning, stacklevel=3,
        )
    with np.errstate(divide="ignore"):
        inv = np.where(bad, 0.0, 1.0 / np.where(bad, 1.0, den))
    return inv, bad


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def construct_ibm_interface_values(ibm, recon, ic, *, det_tol=1e-12,
                                   return_diagnostics=False):
    """Eliminate the interface values for linear interface conditions.

    Substituting the one-sided reconstructions
    ``q_side = alpha_side * c_gamma_side + N_side @ c`` into the two
    interface equations of *ic* gives a 2x2 system per crossing and
    non-spatial layer; its solution expresses the interface values as sparse
    linear functions of the field:

        ``c_gamma_out = H_out @ c + h_out``,
        ``c_gamma_in  = H_in  @ c + h_in``.

    Parameters
    ----------
    ibm : IBM
        Immersed-boundary data from :func:`pymrm.construct_ibm`.
    recon : IBMNormalDerivative
        Reconstruction from
        :func:`pymrm.ibm_recon.construct_ibm_normal_derivative`.
    ic : tuple of dict
        Two interface equations (module docstring).
    det_tol : float, optional
        Relative threshold below which the local 2x2 determinant is treated
        as singular (with a warning naming the crossings).
    return_diagnostics : bool, optional
        Also return a dict with the per-crossing determinant and the
        singular mask.

    Returns
    -------
    H_out, h_out, H_in, h_in
        ``H_*`` are ``csr_array`` of shape ``(n_crossings * ns_size,
        n_cells)``; ``h_*`` are 1-D arrays; rows/entries are ordered
        ``k * ns_size + j`` (matching the ``G_out``/``G_in`` columns of
        :func:`pymrm.apply_ibm`).
    diagnostics : dict, optional
        Only when *return_diagnostics* is true: ``det`` (shape
        ``(n_crossings, ns_size)``) and ``singular`` mask.
    """
    npnt = ibm.n_crossings
    ns = ibm.ns_size
    a, b, d = _ic_coefficients(ic, npnt, ibm.ns_shape)

    alpha_out_full, N_out, alpha_in_full, N_in = \
        construct_ibm_normal_derivative_ops(ibm, recon)
    alpha_out = recon.alpha_out[:, np.newaxis]      # (npnt, 1)
    alpha_in = recon.alpha_in[:, np.newaxis]

    full = (npnt, ns)
    m00 = np.broadcast_to(a[0][0] * alpha_out + b[0][0], full)
    m01 = np.broadcast_to(a[0][1] * alpha_in + b[0][1], full)
    m10 = np.broadcast_to(a[1][0] * alpha_out + b[1][0], full)
    m11 = np.broadcast_to(a[1][1] * alpha_in + b[1][1], full)

    det = m00 * m11 - m01 * m10
    det_inv, singular = _guard_small(det, det_tol, "2x2 elimination matrix")

    minv = ((m11 * det_inv, -m01 * det_inv),
            (-m10 * det_inv, m00 * det_inv))

    def eliminate(row):
        """H and h for the interface value solved from row *row* of M^-1."""
        e_out = np.broadcast_to(
            minv[row][0] * a[0][0] + minv[row][1] * a[1][0], full).ravel()
        e_in = np.broadcast_to(
            minv[row][0] * a[0][1] + minv[row][1] * a[1][1], full).ravel()
        H = (-(_scale_rows(N_out, e_out) + _scale_rows(N_in, e_in))).tocsr()
        h = np.broadcast_to(
            minv[row][0] * d[0] + minv[row][1] * d[1], full).ravel().copy()
        return H, h

    H_out, h_out = eliminate(0)
    H_in, h_in = eliminate(1)

    if return_diagnostics:
        return H_out, h_out, H_in, h_in, {"det": det, "singular": singular}
    return H_out, h_out, H_in, h_in


def apply_ibm_interface(mat, ibm, recon, ic, *, det_tol=1e-12,
                        return_values=False):
    """Apply general linear interface conditions to an operator matrix.

    Combines :func:`pymrm.apply_ibm` (ghost-column folding, source matrices)
    with the interface-value elimination of
    :func:`construct_ibm_interface_values`:

        ``A_final = A_ibm + G_out @ H_out + G_in @ H_in``
        ``g_final = G_out @ h_out + G_in @ h_in``

    Parameters
    ----------
    mat : sparse matrix or array
        Operator matrix of shape ``(n_cells, n_cells)``.
    ibm : IBM
    recon : IBMNormalDerivative
    ic : tuple of dict
        Two interface equations (module docstring).
    det_tol : float, optional
        Passed to :func:`construct_ibm_interface_values`.
    return_values : bool, optional
        Also return ``(H_out, h_out, H_in, h_in)`` for post-processing wall
        values and fluxes.

    Returns
    -------
    A_final : csr_array
        Modified operator matrix.
    g_final : ndarray, shape (n_cells,)
        Interface source (sign convention: *added*, ``value = A @ c + g``).
    values : tuple, optional
        Only when *return_values* is true.
    """
    A_ibm, G_out, G_in = apply_ibm(mat, ibm, return_bc="matrix")
    H_out, h_out, H_in, h_in = construct_ibm_interface_values(
        ibm, recon, ic, det_tol=det_tol)
    A_final = (A_ibm + G_out @ H_out + G_in @ H_in).tocsr()
    g_final = (np.asarray(G_out @ h_out).ravel()
               + np.asarray(G_in @ h_in).ravel())
    if return_values:
        return A_final, g_final, (H_out, h_out, H_in, h_in)
    return A_final, g_final


def construct_ibm_boundary_values(ibm, recon, bc, side="out", *,
                                  det_tol=1e-12):
    """Eliminate the interface value of a single side (immersed Robin BC).

    For a boundary condition on one side only,

        ``a * q_side + b * c_gamma_side = d``

    (outward derivative of that side), the interface value becomes

        ``c_gamma_side = H @ c + h``.

    This covers immersed Neumann/Robin walls where the other region is not
    modelled: pass the returned ``H``/``h`` for this side to the assembly and
    handle the other side as a plain Dirichlet value through
    :func:`pymrm.apply_ibm`, e.g. ::

        A_ibm, G_out, G_in = apply_ibm(A, ibm, return_bc="matrix")
        H, h = construct_ibm_boundary_values(ibm, recon, bc, side="out")
        A_final = (A_ibm + G_out @ H).tocsr()
        g_final = G_out @ h        # solid side: G_in @ values as usual

    Parameters
    ----------
    ibm : IBM
    recon : IBMNormalDerivative
    bc : dict
        Single equation ``{"a": ..., "b": ..., "d": ...}`` (same coefficient
        broadcasting rules as *ic*, but no side pairs).
    side : {'out', 'in'}, optional
        Which side of the interface the condition applies to.
    det_tol : float, optional
        Relative threshold for the local denominator ``a * alpha + b``.

    Returns
    -------
    H : csr_array, shape (n_crossings * ns_size, n_cells)
    h : ndarray, shape (n_crossings * ns_size,)
    """
    npnt = ibm.n_crossings
    ns = ibm.ns_size
    if side not in ("out", "in"):
        raise ValueError(f"side must be 'out' or 'in', got {side!r}")

    a = _coerce_coeff(bc.get("a", 0.0), npnt, ibm.ns_shape, "bc['a']")
    b = _coerce_coeff(bc.get("b", 0.0), npnt, ibm.ns_shape, "bc['b']")
    d = _coerce_coeff(bc.get("d", 0.0), npnt, ibm.ns_shape, "bc['d']")

    ops = construct_ibm_normal_derivative_ops(ibm, recon)
    if side == "out":
        alpha, N = recon.alpha_out[:, np.newaxis], ops[1]
    else:
        alpha, N = recon.alpha_in[:, np.newaxis], ops[3]

    full = (npnt, ns)
    den = np.broadcast_to(a * alpha + b, full)
    den_inv, _ = _guard_small(den, det_tol, "boundary-condition denominator")

    H = -_scale_rows(N, np.broadcast_to(a * den_inv, full).ravel())
    h = np.broadcast_to(d * den_inv, full).ravel().copy()
    return H.tocsr(), h
