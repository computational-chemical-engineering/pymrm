"""Nonlinear-solver utilities used by :mod:`pymrm`."""

import warnings
from dataclasses import dataclass, field
from enum import IntEnum
from typing import NamedTuple, Optional

import numpy as np
from scipy import sparse
from scipy.linalg import solve as dense_solve
from scipy.linalg.lapack import dgetrf as _dgetrf, dgetrs as _dgetrs
from scipy.optimize import Bounds, OptimizeResult
from scipy.sparse import linalg

from .helpers import _csc_abs_matvec

# What ``damping="auto"`` resolves to when neither a callback nor a callable
# linear solver is passed.
_AUTO_DAMPING = "guarded"

# Fixed constants of the globalisation (see the notes of ``newton``).
_LAM_FLOOR = 0.5      # smallest damping a bound alone may impose
_CROSS_RTOL = 1e-8    # a bound crossing shallower than this times the distance is round-off
_LAM_MIN = 1.0e-4     # the damping search ends below this damping
_LAM_NEAR = 0.2       # ... or below this one when the step is within _NEAR tolerances
_NEAR = 100.0         # steps within _NEAR times the tolerance are taken in full
_STAGNATION_WINDOW = 5
_STAGNATION_RATIO = 1.0  # a step no smaller than the smallest of the window
_ARMIJO_C = 1.0e-4
_UNTESTED = float("nan")  # theta of a step no monotonicity test measured (a measured theta can be nan too)
_BACKWARD_WEIGHT = 1.0  # weight of the field scale in the backward error
_GLOBAL_FLOOR = 1.0e-6  # smallest field scale, relative to the largest unknown

# Exceptions that mark an evaluation point as unusable (``newton``'s default).
_RECOVERABLE = (FloatingPointError, OverflowError, ZeroDivisionError)


class _Status(IntEnum):
    """Newton's status codes (``result.status`` stays a plain int)."""

    CONVERGED = 0
    STALLED = 1
    NOT_CONVERGED = 2
    DAMPING_FAILED = 3
    NON_FINITE = 4
    SINGULAR = 5
    SOLVER_FAILED = 6
    BACKWARD_ERROR = 7
    ON_BOUND = 8


_MESSAGES = {
    _Status.CONVERGED: "Converged",
    _Status.STALLED: "Converged to the stagnation floor",
    _Status.NOT_CONVERGED: "Did not converge",
    _Status.DAMPING_FAILED: "Damping fell below its minimum",
    _Status.NON_FINITE: "Non-finite residual",
    _Status.SINGULAR: "Singular Jacobian",
    _Status.SOLVER_FAILED: "Linear solver failed",
    _Status.BACKWARD_ERROR: "Step criterion met but the backward error is above backward_tol",
    _Status.ON_BOUND: "Stopped on a bound",
}


class NewtonBreakdown(np.linalg.LinAlgError):
    """Raised by :func:`newton` when no Newton step can be formed.

    A singular Jacobian (status 5), a non-finite residual or Jacobian that no
    damping can avoid (status 4) or a non-finite Newton step (status 6). It is
    a ``LinAlgError`` and so a ``ValueError``, as the exceptions these cases
    raised up to pymrm 2.5.0. ``result`` holds the
    :class:`scipy.optimize.OptimizeResult` with the status and the last
    iterate ``x``.
    """

    def __init__(self, message, result=None):
        super().__init__(message)
        self.result = result


class KrylovBreakdown(NewtonBreakdown, RuntimeError):
    """An iterative linear solver did not converge (status 6).

    Also a ``RuntimeError``, as raised up to pymrm 2.5.0.
    """


class _SingularJacobian(Exception):
    pass


class _LinearSolverFailed(Exception):
    pass


class _LastEvaluationFailed(Exception):
    pass


def _row_scale(jac):
    """Reciprocal of the largest absolute entry of each row (1 for empty rows)."""
    if sparse.issparse(jac):
        s = abs(jac).max(axis=1)
        s = s.toarray() if sparse.issparse(s) else s
    else:
        s = np.max(np.abs(jac), axis=1)
    s = np.asarray(s, dtype=float).ravel()
    s[s == 0.0] = 1.0
    return 1.0 / s


def _solver_kind(solver):
    """The kind of linear solver ``solver`` names or is."""
    named = solver if isinstance(solver, str) else None     # a solver given by name
    if solver is None or named in ("spsolve", "lu"):
        return "direct"
    if named in ("cg", "bicgstab"):
        return solver
    if named == "splu":
        return "splu"
    if hasattr(solver, "factorize"):
        return "factorize"
    if callable(solver):
        return "callable"
    raise ValueError("Unsupported solver method.")


class _LinearSolver:
    """Builds, per Newton iteration, a ``solve(rhs)`` for the Jacobian."""

    # keyword arguments of scipy.sparse.linalg.splu
    _SPLU_KEYS = ("permc_spec", "diag_pivot_thresh", "relax", "panel_size", "options")

    def __init__(self, solver, kwargs, row_scaling):
        self.kwargs = dict(kwargs or {})
        self.row_scaling = row_scaling
        # legacy (newton sets it for damping="none" without row scaling): factorise as spsolve
        # and scipy.linalg.solve do, so that the plain iteration reproduces pymrm 2.5.0 bit for bit
        self.legacy = False
        self.kind = _solver_kind(solver)
        self.solver = solver
        self.splu_kwargs = {}
        if self.kind == "direct":
            for key, value in self.kwargs.items():
                if key in self._SPLU_KEYS:
                    self.splu_kwargs[key] = value
                elif key != "use_umfpack":
                    warnings.warn(f"newton: lin_solver_kwargs['{key}'] is not an option of "
                                  "the direct solver and is ignored.", RuntimeWarning,
                                  stacklevel=3)   # __init__ <- newton <- the caller

    def factorize(self, jac):
        """Return ``(solve, row_scale)`` for this Jacobian."""
        if self.kind == "direct":
            if sparse.issparse(jac):
                return self._direct_sparse(jac)
            return self._direct_dense(jac)
        if self.kind in ("cg", "bicgstab"):
            return self._iterative(jac), None
        if self.kind == "splu":
            return jac.solve, None
        if self.kind == "factorize":
            factor = self.solver.factorize(jac)
            return factor.solve, None
        solver, kwargs = self.solver, self.kwargs
        return (lambda rhs: solver(jac, rhs, **kwargs)), None

    def _direct_sparse(self, jac):
        options = {"permc_spec": "COLAMD", **self.splu_kwargs}
        if self.legacy and jac.format == "csr":
            # spsolve factors the transpose of a CSR matrix and solves with it
            try:
                lu = linalg.splu(sparse.csc_array(jac.T), **options)
            except RuntimeError as exc:
                raise _SingularJacobian(str(exc)) from exc
            return (lambda rhs: lu.solve(np.asarray(rhs, dtype=float), trans="T")), None
        a = jac if _is_canonical_csc(jac) else sparse.csc_array(jac)
        r = _row_scale(a) if self.row_scaling else None
        if r is not None:
            a = sparse.csc_array(sparse.diags_array(r) @ a)
        try:
            lu = linalg.splu(a, **options)
        except RuntimeError as exc:
            raise _SingularJacobian(str(exc)) from exc
        if r is None:
            return lu.solve, None
        return (lambda rhs: lu.solve(r * rhs)), r

    def _direct_dense(self, jac):
        if self.legacy:
            # scipy.linalg.solve, which detects structure (symmetric, triangular)
            a = np.asarray(jac)
            dense_kwargs = self.kwargs

            def solve(rhs):
                try:
                    return dense_solve(a, np.asarray(rhs), **dense_kwargs)
                except np.linalg.LinAlgError as exc:
                    raise _SingularJacobian(str(exc)) from exc
            return solve, None
        a = np.asarray(jac, dtype=float)
        r = _row_scale(a) if self.row_scaling else None
        if r is not None:
            a = r[:, None] * a
        lu, piv, _ = _dgetrf(a, overwrite_a=False)
        if (lu.diagonal() == 0.0).any():
            raise _SingularJacobian("exactly singular")
        if r is None:
            return (lambda rhs: _dgetrs(lu, piv, rhs)[0]), None
        return (lambda rhs: _dgetrs(lu, piv, r * rhs)[0]), r

    def rhs(self, ev):
        """The right-hand side ``solve`` takes for the evaluation ``ev`` (as returned for a callable solver)."""
        return ev.g_raw if self.kind == "callable" else ev.g

    def _iterative(self, jac):
        if not sparse.issparse(jac):
            raise ValueError(
                f"solver='{self.kind}' requires a sparse Jacobian or a custom solver."
            )
        n = jac.shape[0]
        ilu = linalg.spilu(sparse.csc_array(jac))
        precond = linalg.LinearOperator((n, n), ilu.solve)
        method = linalg.cg if self.kind == "cg" else linalg.bicgstab
        kwargs, name = self.kwargs, self.kind.upper()

        def solve(rhs):
            out, info = method(jac, rhs, M=precond, **kwargs)
            if info != 0:
                raise _LinearSolverFailed(f"{name} did not converge, info={info}")
            return out

        return solve


def _parse_bounds(bounds, shape):
    """Return flat ``(lower, upper)`` arrays, ``None`` for an open side."""
    if bounds is None:
        return None, None
    if isinstance(bounds, Bounds):
        lower, upper = bounds.lb, bounds.ub
    elif isinstance(bounds, tuple) and len(bounds) == 2:
        lower, upper = bounds
    else:
        lower, upper = bounds, None

    def side(value, fill):
        if value is None:
            return None
        arr = np.array(np.broadcast_to(np.asarray(value, dtype=float), shape)).ravel()
        if np.all(arr == fill):
            return None
        return arr

    return side(lower, -np.inf), side(upper, np.inf)


class _Bounds:
    """Fraction-to-the-boundary rule and projection for ``x_new = x - lam dx``."""

    def __init__(self, lower, upper, tau):
        self.lower, self.upper, self.tau = lower, upper, tau
        self.lo_mask = None if lower is None else np.isfinite(lower)
        self.hi_mask = None if upper is None else np.isfinite(upper)
        self.held = None     # components a bound held in the last accepted step
        self.active = lower is not None or upper is not None

    def enforce_initial(self, x):
        """Project an initial guess into the bounds, with a warning."""
        outside = 0
        if self.lower is not None:
            m = self.lo_mask & (x < self.lower)
            outside += int(m.sum())
            x[m] = self.lower[m]
        if self.upper is not None:
            m = self.hi_mask & (x > self.upper)
            outside += int(m.sum())
            x[m] = self.upper[m]
        if outside:
            warnings.warn(f"newton: {outside} components of the initial guess were "
                          "outside the bounds and have been projected onto them.",
                          RuntimeWarning, stacklevel=3)

    def _sides(self, x, dx):
        """Per bound side: sign, mask, bound, distance of ``x`` and overshoot of the full step."""
        full = x - dx
        if self.lower is not None:
            lo = np.where(self.lo_mask, self.lower, 0.0)
            yield 1.0, self.lo_mask, lo, x - lo, lo - full
        if self.upper is not None:
            hi = np.where(self.hi_mask, self.upper, 0.0)
            yield -1.0, self.hi_mask, hi, hi - x, full - hi

    def max_step(self, x, dx):
        """Largest damping that removes at most ``tau`` of the distance to a bound.

        Only components whose full step genuinely crosses their bound limit the
        step; a full step that lands inside the bound, however close, is taken
        (a linear problem with a solution near the bound then converges in one
        step). A crossing shallower than ``_CROSS_RTOL`` times the distance is
        round-off in ``x - dx`` and does not count. Nor does a component that a
        bound held in the previous accepted step: where every cell of an
        extinction tail overshoots its bound on every iteration, it would cap
        every step at ``_LAM_FLOOR`` and only halve the error of all other
        components. It is held again by :meth:`project` if it crosses again.
        Only the last accepted step's holds are excused, not every hold so far.
        """
        lam = 1.0
        for sign, mask, _, dist, over in self._sides(x, dx):
            m = mask & (dist > 0.0) & (over > _CROSS_RTOL * dist)
            if self.held is not None:
                m &= ~self.held
            if np.any(m):
                lam = min(lam, float(np.min(self.tau * dist[m] / (sign * dx[m]))))
        return lam

    def commit(self, x, trial, dx):
        """Record the components :meth:`project` held in the accepted ``trial``.

        A component counts as held when its full step crosses the bound genuinely
        and ``trial`` equals the hold point ``bound + sign (1 - tau) dist`` exactly.
        That covers the components project held and also any that landed exactly on
        the hold point by themselves: the equality test cannot tell them apart.
        ``test_bounds_commit_marks_a_component_that_project_did_not_hold`` in
        ``test/test_known_behaviours.py`` pins this.
        """
        held = np.zeros(x.shape, dtype=bool)
        for sign, mask, bound, dist, over in self._sides(x, dx):
            genuine = mask & (over > _CROSS_RTOL * dist)
            held |= genuine & (trial == bound + sign * (1.0 - self.tau) * dist)
        self.held = held

    def project(self, trial, x, dx):
        """Hold genuinely crossing components ``1 - tau`` of their distance from the bound.

        A round-off crossing, or a landing exactly on the bound, is mirrored
        back inside, at least ``eps`` times the distance from the bound, and
        not counted. A component that already sits on its bound may stay
        there. Returns the number of components held.
        """
        held = 0
        eps = np.finfo(float).eps
        for sign, mask, bound, dist, over in self._sides(x, dx):
            genuine = mask & (over > _CROSS_RTOL * dist)
            hold = genuine & (sign * (trial - bound) < (1.0 - self.tau) * dist)
            trial[hold] = bound[hold] + sign * (1.0 - self.tau) * dist[hold]
            held += int(hold.sum())
            shallow = mask & ~genuine & (dist > 0.0) & (sign * (trial - bound) <= 0.0)
            # reflect through the offset (no 2 * bound, which can overflow)
            offset = np.maximum(sign * (bound[shallow] - trial[shallow]), eps * dist[shallow])
            inside = bound[shallow] + sign * offset
            # an offset below the spacing of floats at the bound rounds back onto it
            step_in = np.nextafter(bound[shallow], x[shallow])
            trial[shallow] = np.where(sign * (inside - step_in) < 0.0, step_in, inside)
        return held


def _field_reference(x, shape, scale, x0_size=0.0):
    """Magnitude of each unknown's field: ``scale``, else the field's largest |x|.

    Fields lie along the last axis (pymrm's convention); a one-dimensional
    state is one field. No field reference is below ``_GLOBAL_FLOOR`` times
    the largest ``|x|``, or the largest ``|x|`` of the initial guess.
    """
    if scale is not None:
        return np.abs(scale)
    ax = np.abs(x).reshape(shape)
    largest = float(ax.max()) if x.size else 0.0
    # The initial guess carries the problem's scale when the solution is ~0.
    floor = _GLOBAL_FLOOR * max(largest, x0_size)
    if len(shape) >= 2:
        # The floor also keeps a single-field state stored without a field
        # axis, (nz, nr), from being judged column by column; a trace field is
        # still judged far below its neighbours.
        ref = np.maximum(ax.reshape(-1, shape[-1]).max(axis=0), floor)
        return np.broadcast_to(ref, shape).reshape(-1)
    return np.full(x.size, max(largest, floor))


def _backward_error(g, jac, x, x_ref):
    """Componentwise backward error with a floor relative to the field scale.

    ``max_i |g_i| / (|J| (|x| + eta x_ref) + |J x - g|)_i`` with
    ``eta = _BACKWARD_WEIGHT``: the measure of
    :func:`pymrm.checks.residual_check`, except that components far below the
    scale of their field (a reaction dead zone, an absent species) are judged
    against that scale instead of against their own size.
    """
    jac = jac if sparse.issparse(jac) else np.asarray(jac, dtype=float)
    v = np.abs(x) + _BACKWARD_WEIGHT * x_ref
    if _is_canonical_csc(jac) and (v.dtype, v.shape) == (np.float64, jac.shape[1:]):
        weighted = _csc_abs_matvec(jac, v)  # abs(jac) @ v, the same floats without the copy
    else:
        weighted = abs(jac) @ v
    denominator = weighted + np.abs(jac @ x - g)
    denominator = np.where(denominator > 0.0, denominator, np.finfo(float).tiny)
    return float(np.max(np.abs(g) / denominator))


def _is_canonical_csc(jac):
    """Whether splu can take ``jac`` as is: a float64 ``csc_array`` without duplicates
    or unsorted indices, so that the copy ``csc_array(jac)`` would change nothing."""
    return type(jac) is sparse.csc_array and jac.dtype == np.float64 and jac.has_canonical_format


def _is_matrix(jac):
    """Whether ``jac`` is a numeric matrix (not a factorisation or user object)."""
    if sparse.issparse(jac):
        return True
    return isinstance(jac, np.ndarray) and jac.dtype.kind in "fiu"


def _rms(v):
    return float(np.sqrt(np.mean(np.square(v))))


def _damping_mode(damping, callback, lin):
    """The damping ``newton`` uses: ``damping``, with ``"auto"`` resolved."""
    if damping == "auto":
        user_hooks = callback is not None or lin.kind == "callable"
        return "none" if user_hooks else _AUTO_DAMPING
    if damping in ("none", "guarded", "monotone", "armijo"):
        return damping
    raise ValueError(f"Unsupported damping {damping!r}.")


def _scale_vector(scale, shape):
    """``scale`` as a flat array of the unknowns' scales (None if not given), checked."""
    if scale is None:
        return None
    d_user = np.array(np.broadcast_to(np.asarray(scale, dtype=float), shape)).ravel()
    if not (np.all(np.isfinite(d_user)) and np.all(d_user > 0.0)):
        raise ValueError("scale must be finite and positive for every unknown.")
    return d_user


class _Eval(NamedTuple):
    """One evaluation: the residual as returned, flat, and the Jacobian."""

    g_raw: object
    g: np.ndarray
    jac: object


class _Step(NamedTuple):
    """One iteration's Newton step and its tests, read-only for the rest of the iteration.

    ``report``: an error at a trial point ends the run with status 2 instead of
    propagating. It is set when the step converged or is the last one, where plain
    Newton would not evaluate again.
    """

    # the step: the evaluation at x, the factorised solve, its row scale, dx and dx / d
    ev: _Eval
    lin_solve: object
    row_scale: object
    dx: np.ndarray
    dy: np.ndarray
    # its tests at x: scaled step norm, step test, near the tolerance, backward-error gate, stagnation
    step_inf: float
    converged: bool
    near: bool
    gate: bool
    stagnant: bool
    be: float
    # whether it may end the run
    last_it: bool
    report: bool


class _Carry(NamedTuple):
    """What an iteration hands to the next, to the verbose line and to ``_exhausted``."""

    ev: _Eval
    step_inf: float
    be: float
    gate_failed: bool
    clipped: int
    lam: float
    theta: float


class _End(NamedTuple):
    """Where the run ends: the evaluation (None if there is none), the last step norm and the backward error."""

    ev: _Eval
    step_norm: float
    be: float


class _Trial(NamedTuple):
    """The point ``x - lam dx`` after ``bnd.project`` and its evaluation (None: unusable or not evaluated)."""

    x: np.ndarray
    ev: _Eval
    lam: float
    clipped: int


class _SearchCtx(NamedTuple):
    """What the tests of a damped search compare with: the rms of the scaled step, the row weights and phi at x."""

    dy_rms: float
    row: object
    phi0: float


class _Found(NamedTuple):
    """The step a damped search found and the last theta it measured."""

    trial: _Trial
    theta: float
    failed: bool         # no usable step: ``trial`` is the fallback and its ``ev`` is None


def _lam_min(near):
    """The damping below which the search ends."""
    # Near the solution a failed test is usually round-off: search briefly.
    return _LAM_NEAR if near else _LAM_MIN


def _search_context(s):
    """The ``_SearchCtx`` of the iteration ``s``, in every damped mode (its warnings are observable)."""
    dy_rms = _rms(s.dy)
    row_scale, jac = s.row_scale, s.ev.jac
    row = row_scale if row_scale is not None else (_row_scale(jac) if _is_matrix(jac) else 1.0)
    phi0 = float(np.linalg.norm(row * s.ev.g))
    return _SearchCtx(dy_rms, row, phi0)


def _phi(row, t):
    """The scaled residual norm at the trial ``t``; None if ``t`` is unusable."""
    return None if t.ev is None else float(np.linalg.norm(row * t.ev.g))


def _stop_status(s, converged):
    """0 if ``converged``, 1 if stalled, None to iterate on (both need the backward-error gate of ``s``)."""
    if not s.gate:
        return None
    if converged:
        return _Status.CONVERGED
    if s.stagnant:
        return _Status.STALLED
    return None


def _is_finite(ev):
    """Whether the residual of ``ev`` and its Jacobian (if a matrix) are finite."""
    finite = np.isfinite(ev.g).all()
    if finite and _is_matrix(ev.jac):
        data = ev.jac.data if sparse.issparse(ev.jac) else ev.jac
        finite = np.isfinite(data).all()
    return finite


def _stalled(window, step_inf):
    """Whether ``step_inf`` is no smaller than the smallest step of a full window."""
    return (len(window) >= _STAGNATION_WINDOW
            and step_inf >= _STAGNATION_RATIO * min(window[-_STAGNATION_WINDOW:]))


def _slide(window, s, use_stagnation):
    """Update the stagnation ``window`` (in place) after the iteration ``s``."""
    if s.stagnant:
        window.clear()      # the backward error did not pass: keep iterating
    elif use_stagnation and s.near:
        # The Newton step norm, damped or not: near the floor the damping
        # test fails on round-off, and a damped step must not reset the window.
        window.append(s.step_inf)
    else:
        window.clear()


class _Criterion(NamedTuple):
    """When a Newton step counts as converged; also the source of every backward error a result reports.

    Built once in :func:`newton`, never written.
    """

    d: np.ndarray            # scale of each unknown (ones without ``scale``)
    tol: object              # the user's tol: |dy|_inf < tol; an array of it selects the relative test (as tol_vec)
    rtol: float              # relative part of the per-unknown limit tol_vec + rtol |x / d|
    tol_vec: object          # tol per unknown, for the relative test (None without it)
    relative: bool           # the per-unknown test with rtol, else |dy|_inf < tol
    zero_step_converges: bool    # False only for damping="none" with backward_tol=None
    # the backward-error gate
    backward_tol: object     # largest backward error a converged step may have (None: no gate)
    field_scale: object      # ``scale`` per unknown, or None: the fields' largest |x|
    x0_size: float           # largest finite |x| of the initial guess
    shape: tuple             # shape of x, for the field reference of the backward error

    def relative_limit(self, x_flat):
        """Per unknown, the largest scaled step the relative test accepts at ``x_flat``."""
        return self.tol_vec + self.rtol * np.abs(x_flat / self.d)

    def step_test(self, dy, step_inf, x_flat):
        """Whether the step ``dy`` (scaled) at ``x_flat`` meets the tolerance, and by how much it exceeds it."""
        tol = self.tol
        if self.relative:
            limit = self.relative_limit(x_flat)
            converged = bool(np.all(np.abs(dy) <= limit))
            with np.errstate(divide="ignore", invalid="ignore"):
                excess = float(np.max(np.where(dy == 0.0, 0.0, np.abs(dy) / limit)))
        else:
            converged = step_inf < tol
            excess = step_inf / tol if tol > 0 else (0.0 if step_inf == 0 else np.inf)
        if self.zero_step_converges:
            converged = converged or step_inf == 0.0
        return converged, excess

    def backward_error(self, ev, x_flat):
        """The backward error of ``ev`` at ``x_flat``; nan unless its Jacobian is a matrix.

        ``ev`` must be the evaluation at ``x_flat``.
        """
        if not _is_matrix(ev.jac):
            return float("nan")
        return _backward_error(ev.g, ev.jac, x_flat,
                               _field_reference(x_flat, self.shape, self.field_scale, self.x0_size))


@dataclass(eq=False, slots=True)
class _Residual:
    """The user's function, its extra arguments and its call count (its only state)."""

    function: object
    args: tuple
    recoverable: tuple       # exceptions that mark a guarded evaluation point unusable
    nfev: int = 0

    def evaluate(self, xv, guard, report):
        """``function(xv, *args)`` as an ``_Eval``, finite or not; None if ``guard`` and it raised a recoverable error.

        With ``report`` any other error becomes ``_LastEvaluationFailed`` (status 2) instead of propagating.
        """
        self.nfev += 1
        skip = self.recoverable if guard else ()
        try:
            g_raw, jac = self.function(xv, *self.args)
        except skip:
            return None
        except Exception as exc:
            if report:
                # Plain Newton would not evaluate here: end the run with status 2.
                raise _LastEvaluationFailed(f"{type(exc).__name__}: {exc}") from exc
            raise
        g_raw = g_raw.toarray() if sparse.issparse(g_raw) else np.array(g_raw, dtype=float)
        g_flat = g_raw.reshape(-1)
        if sparse.issparse(jac) and jac.format not in ("csr", "csc", "coo", "bsr", "dia"):
            jac = jac.tocsc()      # lil, dok: no .data array of the stored values
        return _Eval(g_raw, g_flat, jac)

    def usable(self, xv, guard, report):
        """``evaluate``, None unless the residual and Jacobian are finite."""
        ev = self.evaluate(xv, guard, report)
        if ev is None or not _is_finite(ev):
            return None
        return ev


@dataclass(eq=False, slots=True)
class _NewtonRun:
    """One :func:`newton` solve: its collaborators, its policy, the user's hooks and the iteration state.

    The iteration writes only the state group (``x`` and ``x_flat`` in place, ``nfact``,
    ``history``) and, through its collaborators, ``residual.nfev`` and ``bnd.held``.
    """

    # collaborators: the residual, the convergence criterion, the linear solver, the bounds
    residual: _Residual
    crit: _Criterion
    lin: _LinearSolver
    bnd: _Bounds
    # policy: damping mode, iteration limit, stagnation test, shape and size of x
    mode: str
    maxfev: object
    use_stagnation: bool
    shape: tuple
    n: int
    # the user's hooks
    callback: object
    refresh: object
    verbose: object
    # state: x and its flat view (one array, updated in place), factorisations, history
    x: np.ndarray
    x_flat: np.ndarray
    nfact: int = 0
    history: list = field(default_factory=list)

    # -- the end of the run -----------------------------------------------------------------

    def finish(self, status, end, extra=""):
        """The result of the run ending with ``status`` at ``end``."""
        ev = end.ev
        g_raw = None if ev is None else ev.g_raw
        message = _MESSAGES[status] + (f": {extra}" if extra else "")
        return OptimizeResult(x=self.x, success=status in (_Status.CONVERGED, _Status.STALLED),
                              status=int(status), message=message,
                              nit=len(self.history), nfev=self.residual.nfev, nfact=self.nfact, fun=g_raw,
                              step_norm=end.step_norm, backward_error=end.be, history=self.history)

    def _breakdown(self, status, end, extra, kind=NewtonBreakdown):
        """The exception that ends a run without a usable Newton step (statuses 4, 5 and 6), with its result.

        These raise; callers catch them as ValueError, LinAlgError or RuntimeError.
        """
        result = self.finish(status, end, extra)
        return kind(result.message, result)

    # -- the start of the run ---------------------------------------------------------------

    def solve(self, guard_guess):
        """Evaluate at the initial guess and iterate; the result of :func:`newton`."""
        residual = self.residual
        if guard_guess:
            # The guess is a prediction (continuation, pseudo_transient): one of
            # ``recoverable`` there marks it unusable, like a non-finite residual.
            try:
                ev = residual.evaluate(self.x, guard=False, report=False)
            except residual.recoverable as exc:
                raise self._breakdown(_Status.NON_FINITE, _End(None, float("nan"), float("nan")),
                                      f"{type(exc).__name__} at the initial guess ({exc})")
        else:
            ev = residual.evaluate(self.x, guard=False, report=False)
        if not _is_finite(ev):
            raise self._breakdown(_Status.NON_FINITE, _End(ev, float("nan"), float("nan")), "at the initial guess")
        return self.iterate(ev)

    # -- one iteration ----------------------------------------------------------------------

    def iterate(self, ev):
        """Newton steps from ``ev`` (at x) until a stop: the result.

        Two orders, by ``mode``. Plain (``"none"``, the 2.5.0 iteration) decides the absolute
        test on dy at x, takes the full or bound-shortened step, then makes the relative test,
        which reads x, again at the new x. Damped decides at x, then stops with the full step
        or searches. Both call the callback and refresh after every step they take, the last
        included, and before the next evaluation.
        """
        carry = _Carry(ev, float("nan"), float("nan"), False, 0, 1.0, _UNTESTED)
        window = []          # step norms of the recent full steps at the floor
        maxfev = int(self.maxfev)
        for it in range(maxfev):
            s = self._newton_step(carry.ev, carry.step_inf, it == maxfev - 1, window)
            out = self._plain_step(s) if self.mode == "none" else self._damped_step(s, carry.lam, it)
            if isinstance(out, OptimizeResult):
                return out
            # Report the iteration and update the stagnation window.
            if self.verbose:
                print(f"newton it={it:3d} |dx|={out.step_inf:.3e} lam={out.lam:.3g} "
                      f"theta={out.theta:.3g} clipped={out.clipped} nfev={self.residual.nfev}")
            carry = out
            _slide(window, s, self.use_stagnation)
        return self._exhausted(carry)

    def _factorize(self, ev, step_inf):
        """Factorise the Jacobian at ``ev`` and solve for the Newton step: ``(solve, row_scale, dx)``.

        A singular Jacobian (5), a failed linear solve or a non-finite step (6) raise a breakdown
        reporting ``step_inf``, the last step norm.
        """
        try:
            solve, row_scale = self.lin.factorize(ev.jac)
            self.nfact += 1
            dx = np.asarray(solve(self.lin.rhs(ev)), dtype=float).reshape(-1)
        except _SingularJacobian as exc:
            raise self._breakdown(_Status.SINGULAR, _End(ev, step_inf, float("nan")), str(exc))
        except _LinearSolverFailed as exc:
            raise self._breakdown(_Status.SOLVER_FAILED, _End(ev, step_inf, float("nan")), str(exc), KrylovBreakdown)
        if not np.isfinite(dx).all():
            raise self._breakdown(_Status.SOLVER_FAILED, _End(ev, step_inf, float("nan")), "non-finite Newton step")
        return solve, row_scale, dx

    def _newton_step(self, ev, prev_step, last_it, window):
        """The Newton step at ``ev``, its norm, convergence and stagnation tests and backward error."""
        crit = self.crit
        lin_solve, row_scale, dx = self._factorize(ev, prev_step)     # prev_step: the norm reported if it fails
        dy = dx / crit.d
        step_inf = float(np.abs(dy).max()) if self.n else 0.0
        converged, excess = crit.step_test(dy, step_inf, self.x_flat)
        near = converged or (self.refresh is None and excess < _NEAR)

        # The verdict uses the residual at x, before the step: no evaluation
        # beyond those of plain Newton. Near a solution the step only improves it.
        # With the relative test the backward error is always measured: plain mode re-tests
        # convergence at the new x and needs a measured gate there.
        be = float("nan")
        measured = near or last_it or crit.relative
        if measured:
            be = crit.backward_error(ev, self.x_flat)
        # An unmeasured be is nan and passes the gate. That is never decisive: the gate is read only with
        # converged or stagnant (_stop_status, gate_failed); both imply near, or the relative test, which
        # always measures.
        backward_tol = crit.backward_tol
        gate = backward_tol is None or np.isnan(be) or be <= backward_tol
        stagnant = self.use_stagnation and near and _stalled(window, step_inf)
        return _Step(ev, lin_solve, row_scale, dx, dy, step_inf, converged, near, gate, stagnant, be, last_it,
                     converged or last_it)

    def _plain_step(self, s):
        """Plain (or bound-shortened) step, callback and refresh, then stop or evaluate at the new x.

        The callback and refresh come before the next evaluation, as the callers of the
        undamped iteration expect. The result, or the ``_Carry``.
        """
        bnd = self.bnd
        lam = 1.0
        if not s.near and bnd.active:
            lam = max(_LAM_FLOOR, bnd.max_step(self.x_flat, s.dx))
        t = self._trial(s, lam, evaluate=False)
        self._take(s, t, _UNTESTED)
        converged = s.converged
        crit = self.crit
        if crit.relative and s.step_inf > 0.0:
            # The relative test uses x after the step (and the callback).
            converged = bool(np.all(np.abs(s.dy) <= crit.relative_limit(self.x_flat)))
        status = _stop_status(s, converged)
        if status is not None:
            return self._final(s, status, t.clipped)
        gate_failed = converged and not s.gate
        if s.last_it:
            # no evaluation after the last step: the result reports the residual at x before it
            return self._exhausted(_Carry(s.ev, s.step_inf, s.be, gate_failed, t.clipped, t.lam, _UNTESTED))
        ev = self.residual.usable(self.x, guard=False, report=False)
        if ev is None:
            raise self._breakdown(_Status.NON_FINITE, _End(s.ev, s.step_inf, float("nan")), "after a full step")
        return _Carry(ev, s.step_inf, s.be, gate_failed, t.clipped, t.lam, _UNTESTED)

    def _damped_step(self, s, lam_prev, it):
        """Stop, or search a damped step, take it and evaluate there: the result or the ``_Carry``."""
        status = _stop_status(s, s.converged)
        if status is not None:
            # Take the converged or stalled step without evaluating there, as plain Newton does
            # (status 8 only if that step is held by a bound).
            t = self._trial(s, 1.0, evaluate=False)
            self._take(s, t, _UNTESTED)
            return self._final(s, status, t.clipped)
        try:
            found = self._search(s, lam_prev)
            t = found.trial
            if found.failed:
                return self._damping_failed(s, t.clipped, it)
            ev = t.ev
            if self._take(s, t, found.theta):
                ev = self.residual.usable(self.x, guard=True, report=s.report)
                if ev is None:
                    raise self._breakdown(_Status.NON_FINITE, _End(t.ev, s.step_inf, float("nan")),
                                          "after the callback or refresh")
        except _LastEvaluationFailed as exc:
            return self.finish(_Status.NOT_CONVERGED, _End(s.ev, s.step_inf, float("nan")),
                               f"the residual at the last trial point raised {exc}")
        return _Carry(ev, s.step_inf, s.be, s.converged and not s.gate, t.clipped, t.lam, found.theta)

    # -- the damped search ------------------------------------------------------------------

    def _search(self, s, lam_prev):
        """The damped step of the iteration ``s``: a ``_Found``.

        Guarded mode takes the full step (shortened only by the bounds far from the
        solution) if its residual is usable, else runs the monotone search from a quarter
        of it. The monotone and Armijo searches start from four times the last damping, at
        most 1 and lam_max.
        """
        bnd = self.bnd
        lam_max = max(_LAM_FLOOR, bnd.max_step(self.x_flat, s.dx)) if bnd.active else 1.0
        if self.mode == "guarded":
            full = 1.0 if s.near else lam_max
            t = self._trial(s, full)
            if t.ev is not None:
                return _Found(t, _UNTESTED, False)
            # An unusable full step at lam_max need not be evaluated again by the fallback.
            return self._monotone_search(s, min(0.25 * full, lam_max), lam_max, t if full == lam_max else None)
        lam = min(min(1.0, 4.0 * lam_prev), lam_max)
        if self.mode == "armijo":
            return self._armijo_search(s, lam, lam_max)
        return self._monotone_search(s, lam, lam_max, None)

    def _monotone_search(self, s, lam, lam_max, at_lam_max):
        """Shorten lam until the simplified Newton step contracts (``_contraction``), else the fallback.

        An unusable trial quarters lam. ``at_lam_max``: the trial at ``lam_max`` if already made.
        """
        ctx = _search_context(s)
        lam_min = _lam_min(s.near)
        theta = _UNTESTED
        while lam >= lam_min:
            t = self._trial(s, lam)
            if lam == lam_max:
                at_lam_max = t
            if t.ev is None:
                lam *= 0.25
            else:
                theta, lam_next = self._contraction(s, t, ctx.dy_rms)
                if lam_next is None:
                    return _Found(t, theta, False)
                lam = lam_next
        t = self._fallback(s, lam_max, at_lam_max, None)
        return _Found(t, theta, t.ev is None)

    def _contraction(self, s, t, dy_rms):
        """Theta of the simplified Newton step at ``t`` and the next lam; None if ``t`` passes the test.

        Only damped modes get here, so ``lin.legacy`` (mode ``"none"``) is off and the
        legacy dense ``_SingularJacobian`` cannot arise from ``s.lin_solve``.
        """
        lam = t.lam
        try:
            dbar = np.asarray(s.lin_solve(self.lin.rhs(t.ev)), dtype=float).reshape(-1) / self.crit.d
        except _LinearSolverFailed:
            dbar = np.full(self.n, np.inf)    # no test possible: reject the trial
        theta = _rms(dbar) / dy_rms if dy_rms > 0.0 else 0.0
        if not np.isfinite(theta):
            return theta, lam * 0.5
        if theta <= 1.0 - lam / 4.0:
            return theta, None
        mu = 0.5 * dy_rms * lam**2 / max(_rms(dbar - (1.0 - lam) * s.dy), 1.0e-300)
        return theta, min(0.5 * lam, max(mu, 0.1 * lam))

    def _armijo_search(self, s, lam, lam_max):
        """Halve lam until the scaled residual decreases enough (Armijo), else the fallback.

        An unusable trial quarters lam. ``best``: the trial with the smallest scaled
        residual below that at x.
        """
        ctx = _search_context(s)
        lam_min = _lam_min(s.near)
        at_lam_max, best, best_phi = None, None, ctx.phi0
        while lam >= lam_min:
            t = self._trial(s, lam)
            phi = _phi(ctx.row, t)
            if lam == lam_max:
                at_lam_max = t
            if phi is None:
                lam *= 0.25
                continue
            if phi <= (1.0 - _ARMIJO_C * lam) * ctx.phi0:
                return _Found(t, _UNTESTED, False)
            if phi < best_phi:
                best, best_phi = t, phi
            lam *= 0.5
        t = self._fallback(s, lam_max, at_lam_max, best)
        return _Found(t, _UNTESTED, t.ev is None)

    def _fallback(self, s, lam_max, at_lam_max, best):
        """The step when no trial passed: ``best`` evaluated again if usable, else the bounded full step.

        The full step's ``ev`` is None unless its residual is finite; made before, it is
        evaluated again (the model may hold a later trial's state).
        """
        if best is not None:
            t = self._reevaluate(s, best)
            if t.ev is not None:
                return t
        if at_lam_max is None:
            return self._trial(s, lam_max)
        if at_lam_max.ev is None:
            return at_lam_max
        return self._reevaluate(s, at_lam_max)

    def _trial(self, s, lam, evaluate=True):
        """The point ``x - lam dx`` held inside the bounds, and (if ``evaluate``) its usable evaluation."""
        x_flat, dx, bnd = self.x_flat, s.dx, self.bnd
        trial = x_flat - lam * dx
        clipped = bnd.project(trial, x_flat, dx) if bnd.active else 0
        ev = self.residual.usable(trial.reshape(self.shape), guard=True, report=s.report) if evaluate else None
        return _Trial(trial, ev, lam, clipped)

    def _reevaluate(self, s, t):
        """The trial ``t`` evaluated again (restoring the model state)."""
        return _Trial(t.x, self.residual.usable(t.x.reshape(self.shape), guard=True, report=s.report), t.lam, t.clipped)

    # -- taking a step and the end of the iteration ----------------------------------------

    def _take(self, s, t, theta):
        """Take the step to the trial ``t``, then the callback and refresh; whether one of them ran."""
        x_flat, bnd = self.x_flat, self.bnd
        if bnd.active:
            bnd.commit(x_flat, t.x, s.dx)
        x_flat[:] = t.x
        self.history.append({"step_norm": s.step_inf, "damping": t.lam, "theta": theta,
                             "clipped": t.clipped})
        stale = False
        if self.callback is not None:
            # The callback may change x or model state: re-evaluate after it.
            self.callback(self.x, s.ev.g_raw)
            stale = True
        if self.refresh is not None:
            self.refresh(self.x)
            stale = True
        return stale

    def _final(self, s, status, clipped):
        """The result of a converged (0) or stalled run (1, or 8 if its last step is held by a bound)."""
        step_inf = s.step_inf
        if status == _Status.CONVERGED:
            return self.finish(_Status.CONVERGED, _End(s.ev, step_inf, s.be))
        if clipped:
            return self.finish(_Status.ON_BOUND, _End(s.ev, step_inf, s.be), f"{clipped} components held by a bound")
        return self.finish(_Status.STALLED, _End(s.ev, step_inf, s.be),
                           f"|dx| = {step_inf:.2e} (tol {np.max(self.crit.tol):.1e})")

    def _damping_failed(self, s, clipped, it):
        """The result when no damped step was found (status 3)."""
        extra = f"{_lam_min(s.near):g} at iteration {it}"
        if clipped:
            extra += f"; {clipped} components held by a bound"
        return self.finish(_Status.DAMPING_FAILED,
                           _End(s.ev, s.step_inf, self.crit.backward_error(s.ev, self.x_flat)), extra)

    def _exhausted(self, carry):
        """The result after ``maxfev`` iterations without convergence."""
        ev, step_inf, be = carry.ev, carry.step_inf, carry.be
        if self.mode != "none":
            be = self.crit.backward_error(ev, self.x_flat)     # ev is at x after a damped step
        if carry.gate_failed:
            return self.finish(_Status.BACKWARD_ERROR, _End(ev, step_inf, be), f"{be:.2e} after maxfev iterations")
        if carry.clipped:
            return self.finish(_Status.ON_BOUND, _End(ev, step_inf, be), f"{carry.clipped} components held by a bound")
        return self.finish(_Status.NOT_CONVERGED, _End(ev, step_inf, be))


def newton(
    function,
    initial_guess,
    args=(),
    tol=1.49012e-08,
    maxfev=100,
    solver=None,
    lin_solver_kwargs=None,
    callback=None,
    rtol=0.0,
    *,
    damping="auto",
    bounds=None,
    tau=0.99,
    scale=None,
    refresh=None,
    backward_tol=1e-8,
    stagnation=False,
    row_scaling=False,
    recoverable=_RECOVERABLE,
    verbose=0,
    _guard_guess=False,
):
    """Solve ``function(x) = 0`` with a (damped) Newton iteration.

    Every iteration solves ``J dx = g`` with a direct sparse LU factorisation
    and updates ``x <- x - lam dx``. With the default options the step length
    ``lam`` is 1 (plain Newton) unless the full step gives a non-finite
    residual, and convergence requires both a small step (``tol``) and a small
    backward error (``backward_tol``). For hard problems, ``damping``,
    ``bounds`` and ``scale`` make the iteration robust far from the solution;
    see the Notes.

    Parameters
    ----------
    function : callable
        ``function(x, *args) -> (residual, jacobian)``. ``x`` has the shape of
        ``initial_guess``; the residual is flattened internally. The Jacobian
        is a scipy sparse matrix or array, a dense array, or, with
        ``solver='splu'``, an already factorised ``SuperLU`` object.
    initial_guess : numpy.ndarray
        Starting point; it is copied, not modified.
    args : tuple, optional
        Extra positional arguments passed to ``function``.
    tol : float or numpy.ndarray, optional
        Stopping tolerance on the Newton step. With the default ``rtol=0`` and
        a scalar ``tol`` the step test is ``max |dx| < tol``. This is an
        ABSOLUTE criterion: for unknowns much smaller than ``tol`` (trace
        concentrations) it can stop after one step with a wrong answer. Scale
        the unknowns to order one (``scale``), use ``tol=0`` with ``rtol``, or
        set ``backward_tol``. An array must broadcast to the unknowns. With
        ``scale`` the test applies to the scaled step ``dx / scale``.
    maxfev : int, optional
        Maximum number of Newton iterations (Jacobian factorisations), not of
        function evaluations; ``nfev`` in the result counts those.
    solver : None, str, callable or object, optional
        Linear solver for ``J dx = g``.

        * ``None``, ``'spsolve'`` or ``'lu'``: direct LU at every size,
          ``scipy.sparse.linalg.splu`` (COLAMD ordering) for sparse Jacobians
          and ``scipy.linalg.lu_factor`` for dense ones. The factorisation is
          reused for the damping test. ``lin_solver_kwargs`` go to ``splu``.
        * ``'cg'``, ``'bicgstab'``: ILU-preconditioned Krylov solvers, only on
          request; ``lin_solver_kwargs`` go to the Krylov solver.
        * ``'splu'``: ``function`` returns an already factorised ``SuperLU``
          object instead of a matrix; its ``.solve()`` is used.
        * an object with a ``factorize(jac)`` method returning an object with
          ``.solve(rhs)``: a user factorisation (for example a Schur
          complement) that takes part in the damping test.
        * a callable ``solver(jac, rhs, **lin_solver_kwargs) -> dx``.
    lin_solver_kwargs : dict, optional
        Keyword arguments forwarded to the linear solver.
    callback : callable, optional
        ``callback(x, residual)`` after each accepted step, with ``residual``
        the residual before the step. It may modify ``x`` in place (for
        example :func:`clip_approach`); the change is kept.
    rtol : float, optional
        Relative stopping tolerance. When ``rtol > 0`` or ``tol`` is an array,
        the step test is ``abs(dx) <= tol + rtol * abs(x)`` for every
        component. ``tol=0, rtol=1e-8`` gives a purely relative criterion.
    damping : {'auto', 'none', 'guarded', 'monotone', 'armijo'}, optional
        Globalisation of the step.

        * ``'none'``: full Newton steps (shortened only by ``bounds``).
        * ``'guarded'``: full steps, unless the residual at the full step is
          not finite or its evaluation raises one of ``recoverable``; only
          then is the step shortened by the monotone search. Where plain
          Newton works it takes exactly the same steps.
        * ``'monotone'``: Deuflhard's natural monotonicity test. A step length
          ``lam`` is accepted when the simplified Newton correction at the
          trial point, ``J(x)^{-1} g(x - lam dx)``, is smaller than
          ``(1 - lam/4)`` times ``dx`` (root mean square, scaled by
          ``scale``); otherwise ``lam`` is reduced. Independent of how the
          residual rows are scaled; costs one back substitution per trial.
          When no step length above 1e-4 passes (0.2 near the solution), the
          full step is taken if its residual is finite: the test cannot see
          past a turning point, which full steps sometimes jump.
        * ``'armijo'``: backtracking until the row-equilibrated residual norm
          decreases by the Armijo condition, with the same fallback. Use it
          when the simplified correction is unavailable or meaningless.
        * ``'auto'`` (default): ``'guarded'``, but ``'none'`` whenever a
          ``callback`` or a callable ``solver`` is given, since those may
          change the step after it is taken.

        Measured on 84 Arrhenius (Weisz-Hicks) pellets from three starts:
        ``'none'`` 60 converged, ``'guarded'`` 62 at the same cost,
        ``'monotone'`` 69 at about twice the function evaluations. On
        half-order kinetics with a dead zone ``'monotone'`` with regularised
        kinetics converged in every case, plain Newton in a third.
    bounds : float, array, tuple, or scipy.optimize.Bounds, optional
        ``(lower, upper)``; each side ``None``, a scalar, or an array that
        broadcasts against the shape of ``initial_guess`` (shape ``(nc,)``
        gives one bound per field, ``(nz, 1)`` one per cell). Use ``-np.inf``
        or ``np.inf`` for unbounded components. A single scalar or array means
        a lower bound. A full step that crosses a bound is shortened so that
        it removes at most ``tau`` of the distance of that component to its
        bound (with a floor of 0.5 on ``lam``, below which the crossing
        components are held individually). A full step that lands inside the
        bounds is taken, however close to a bound, so a solution near a bound
        still converges quadratically. A component may sit on its bound; an
        initial guess outside is projected, with a warning. Use bounds where
        the residual is undefined (``T > 0``, arguments of ``log`` or
        ``sqrt``), not to enforce physical positivity: see Notes. Where the
        residual is singular at the bound (``log``), prefer
        ``damping="monotone"``: a step may land very close to the bound.
    tau : float, optional
        Fraction-to-the-boundary parameter of ``bounds``.
    scale : float or numpy.ndarray, optional
        Typical magnitude of the unknowns, broadcast against
        ``initial_guess``. The step test, the damping test and the stagnation
        rule are applied to ``x / scale``.
    refresh : callable, optional
        ``refresh(x)`` after every accepted step, for quantities the caller
        updates outside the Jacobian (lagged velocities, properties). The
        residual is re-evaluated afterwards. The damping test is then
        approximate, and the stagnation rule is off.
    backward_tol : float or None, optional
        If not None (default ``1e-8``), the step test is necessary but not
        sufficient (unless the Jacobian is not a matrix, so that the backward
        error cannot be formed): the backward error of the iterate the step is
        taken from must also be below ``backward_tol``, else the iteration
        continues. The final Newton step then only improves it, and no
        evaluation is spent beyond those of plain Newton. The backward error is
        ``max_i |g_i| / (|J| (|x| + x_ref) + |J x - g|)_i``, the relative
        change of the equation coefficients that would make ``x`` exact, with
        ``x_ref`` the magnitude of each unknown's field: ``scale`` if given,
        else the largest ``|x|`` of that field (fields along the last axis; a
        one-dimensional state is one field). Components far below their
        field's magnitude, such as a reaction dead zone, are thus judged
        against the field, while a field that is small everywhere (a trace
        species) is judged against itself, down to ``1e-6`` of the largest
        unknown. It is always reported in the result. For a flat state that
        mixes units, pass ``scale``. ``None`` restores the plain step test.
    stagnation : bool, optional
        Stop at the stagnation floor: when the step test cannot be met because
        round-off, or switching terms such as ``max(c, 0)`` and upwinding, keep
        the steps from shrinking, a Newton step within 100 tolerances that is
        no smaller than the smallest of the five before it ends the iteration
        with success, provided ``backward_tol`` passes; otherwise the iteration
        continues. Steps the damping shortened count too: at the floor the
        damping test fails on round-off. A converging iteration, however slowly, makes a new smallest
        step every time and is never stopped. Off by default: code written for
        the old iteration may rely on a ``tol`` below the floor failing.
    row_scaling : bool, optional
        Divide each row of the Jacobian by its largest entry before the direct
        LU factorisation. Improves pivoting when rows have different units.
        Off by default: it changes the pivot order and with it the round-off,
        and continuations that pass close to a turning point with a
        tolerance near round-off then end on a different branch (measured on
        a gallery page).
    recoverable : tuple of exception types, optional
        Exceptions that mark a damped trial point as unusable: the step is
        shortened. Non-finite residuals are treated the same way. Exceptions
        at the initial guess, and in undamped iterations, propagate.
    verbose : int, optional
        ``1`` prints one line per iteration.

    Returns
    -------
    scipy.optimize.OptimizeResult
        ``x`` (shape of ``initial_guess``), ``success``, ``status``,
        ``message``, ``nit`` (Newton iterations), ``nfev`` (function
        evaluations), ``nfact`` (factorisations), ``fun`` (the residual of
        the last evaluated iterate, as returned by ``function``: for a
        converged solve the iterate before the final step, which ``x`` only
        improves; evaluate ``function(x)`` for the residual at ``x``),
        ``step_norm`` (infinity norm of the last Newton step),
        ``backward_error`` (of the same iterate as ``fun``; ``nan`` when the
        Jacobian is not a matrix), and ``history``, a list with per
        iteration ``step_norm``, ``damping``, ``theta`` (monotonicity ratio)
        and ``clipped`` (components held by a bound).

        ``status``: 0 converged; 1 converged to the stagnation floor;
        2 ``maxfev`` reached; 3 damping fell below its minimum; 7 step test
        met but ``backward_error > backward_tol``; 8 stopped on a bound (the
        bound is likely cutting off the solution). Statuses 4 (non-finite
        residual or Jacobian that no damping avoids), 5 (singular Jacobian)
        and 6 (non-finite Newton step, or an iterative linear solver that did
        not converge) are raised as :class:`NewtonBreakdown`, whose ``result``
        holds this object.

    Raises
    ------
    ValueError
        If ``solver``, ``damping`` or ``scale`` is not supported.
    NewtonBreakdown
        When no Newton step can be formed (statuses 4 to 6). It is a
        ``numpy.linalg.LinAlgError`` and so a ``ValueError``, as the
        exceptions these cases raised up to 2.5.0; :class:`KrylovBreakdown`
        (an iterative solver failed) is also a ``RuntimeError``.

    Notes
    -----
    **When plain Newton fails.** Far from the solution a full step can leave
    the region where the residual is meaningful (a negative concentration or
    temperature), and the iteration ends in a non-finite residual or
    diverges. ``damping='guarded'`` handles the non-finite case at no cost
    elsewhere. Use ``damping='monotone'`` when full steps overshoot without
    producing non-finite values; add ``scale`` when unknowns of
    different units are mixed (it makes the step and damping tests weigh
    them equally), and ``bounds`` only for unknowns that the residual cannot
    accept beyond a limit. Where a solution depends on a parameter, warm
    starts along that parameter (continuation) remain the most reliable
    route; damping does not choose between multiple solutions.

    **Kinetics near zero concentration.** A hard lower bound at zero often
    hurts convergence: the discrete solution may itself be slightly negative
    near steep fronts, and ``max(c, 0)`` in a rate makes its derivative zero
    below zero, so the Jacobian loses the term that pulls ``c`` back. Prefer
    rates that are smooth through zero and first order near it, for example
    ``k c (c**2 + eps**2)**((n - 1)/2)`` instead of ``k c**n`` with ``n < 1``,
    or ``c / (1 + K abs(c))`` for Langmuir-Hinshelwood forms, with ``eps`` a
    small fraction of a reference concentration (check that results do not
    change when it is reduced tenfold). A guard bound below zero, such as
    ``bounds=(-1e-3 * c_ref, None)``, then only stops wild overshoots.

    **Convergence verdict.** The step test is absolute. A trace unknown much
    smaller than ``tol`` passes it immediately; the default
    ``backward_tol=1e-8`` then keeps the iteration going until the residual
    is also small relative to the equations, without changing the meaning of
    ``tol``. :func:`pymrm.checks.residual_check` gives a stricter, purely
    componentwise verdict after the fact.

    **Changes in 3.0.0.** Before, ``newton`` took plain full steps even
    into a non-finite residual, stopped on the step test alone, and switched
    to ILU-BiCGSTAB above 50 000 unknowns. Pass ``damping="none",
    backward_tol=None`` for the old iteration; it then
    takes the same steps, evaluations and results as before. A singular
    Jacobian, a non-finite residual or step, or an iterative linear solver
    that does not converge still raise, now as :class:`NewtonBreakdown`
    with the result attached.
    """
    x = np.array(initial_guess, dtype=float, order="C")
    shape = x.shape
    n = x.size
    x_flat = x.reshape(-1)  # view: updates of x_flat update x
    with np.errstate(invalid="ignore"):
        x0_size = float(np.nanmax(np.abs(x_flat))) if n and np.any(np.isfinite(x_flat)) else 0.0

    lin = _LinearSolver(solver, lin_solver_kwargs, row_scaling)

    mode = _damping_mode(damping, callback, lin)
    # The damping="none" compatibility path: there, without row scaling, lin.legacy factorises as
    # pymrm 2.5.0 did, and with backward_tol=None zero_step_converges is False, so a zero step
    # converges only if it passes the step test, as before 3.0.0.
    lin.legacy = mode == "none" and not row_scaling
    zero_step_converges = mode != "none" or backward_tol is not None

    d_user = _scale_vector(scale, shape)
    d = np.ones(n) if d_user is None else d_user
    relative = rtol > 0 or not np.isscalar(tol)
    tol_vec = np.array(np.broadcast_to(np.asarray(tol, dtype=float), shape)).ravel() \
        if relative else None
    bnd = _Bounds(*_parse_bounds(bounds, shape), tau)
    if bnd.active:
        bnd.enforce_initial(x_flat)
    use_stagnation = stagnation and refresh is None
    crit = _Criterion(d=d, tol=tol, rtol=rtol, tol_vec=tol_vec, relative=relative,
                      zero_step_converges=zero_step_converges, backward_tol=backward_tol,
                      field_scale=d_user, x0_size=x0_size, shape=shape)

    run = _NewtonRun(residual=_Residual(function, args, recoverable), x=x, x_flat=x_flat,
                     shape=shape, n=n, crit=crit, lin=lin, mode=mode,
                     bnd=bnd, use_stagnation=use_stagnation, maxfev=maxfev,
                     callback=callback, refresh=refresh, verbose=verbose)
    return run.solve(_guard_guess)


def _mass_matrix(mass, shape):
    """Sparse matrix of the pseudo-time accumulation term."""
    if sparse.issparse(mass):
        return sparse.csc_array(mass)
    diag = np.array(np.broadcast_to(np.asarray(mass, dtype=float), shape)).ravel()
    return sparse.diags_array(diag).tocsc()


def _attempt(function, x, args, options):
    """newton, with a breakdown returned as an unsuccessful result.

    The guess is a prediction (a secant, a pseudo-time state), so one of
    newton's ``recoverable`` exceptions raised by its evaluation marks an
    unusable point, as a non-finite residual does: status 4, and the caller
    shortens its step. Exceptions later in the solve propagate as in newton.
    """
    try:
        return newton(function, x, args, **options, _guard_guess=True)
    except NewtonBreakdown as exc:
        return exc.result


_EASY_NIT = 3          # a pseudo-time step whose Newton solve took at most this many iterations was easy
_MIN_GROWTH = 2.0      # after an easy step dt grows at least by this factor
_STEP_TRIES = 20       # tries of one pseudo-time step before the march fails
_DT_SHRINK = 0.25      # factor on dt after a failed try


class _March(NamedTuple):
    """State of a :func:`pseudo_transient` march between its stages."""

    x: np.ndarray
    dt: float
    history: list
    g_norm: Optional[float] = None       # residual norm at x
    g_prev: Optional[float] = None       # residual norm before the last pseudo-time step
    g_attempt: Optional[float] = None    # residual norm at the last steady attempt
    steady_result: object = None      # the last steady newton result


class _Stage(NamedTuple):
    """A stage of the march: the result if the march ends there, and the state after it."""

    end: object
    state: _March


@dataclass(eq=False)
class _PseudoTransient:
    """The settings of a :func:`pseudo_transient` march and its stages; the march state is a :class:`_March`."""

    function: object
    args: tuple
    mass_matrix: object
    inner: dict
    steady: dict
    dt_max: float
    max_steps: object
    growth: float
    steady_ratio: float
    steady_every: object
    verbose: object

    def result(self, success, message, nsteps, s):
        xr = s.steady_result.x if (success and s.steady_result is not None) else s.x
        return OptimizeResult(x=xr, success=success, message=message, nsteps=nsteps, history=s.history,
                              steady=s.steady_result)

    def march(self, x, dt):
        """Pseudo-time steps from ``x`` with steady Newton attempts; the result of :func:`pseudo_transient`."""
        s = _March(x=x, dt=dt, history=[])
        for step in range(int(self.max_steps) + 1):
            end, s = self._steady_check(step, s)
            if end is None and step != self.max_steps:
                end, s = self._time_step(step, s)
            if end is not None:
                return end
        return self.result(False, "max_steps reached without a converged steady solve", int(self.max_steps), s)

    def _steady_due(self, step, g_norm, g_attempt):
        """Whether to try the steady Newton solve at this step."""
        if g_attempt is None or g_norm <= self.steady_ratio * g_attempt:
            return True
        return step == self.max_steps or (self.steady_every and step % self.steady_every == 0)

    def _steady_check(self, step, s):
        """Residual norm at ``s.x`` and, when due, a steady solve; the result if the march ends."""
        g_now, _ = self.function(s.x, *self.args)
        s = s._replace(g_norm=float(np.linalg.norm(np.asarray(g_now, dtype=float).ravel())))
        if not np.isfinite(s.g_norm):
            return _Stage(self.result(False, "non-finite residual during the pseudo-time march", step, s), s)
        if not self._steady_due(step, s.g_norm, s.g_attempt):
            return _Stage(None, s)
        s = s._replace(g_attempt=s.g_norm, steady_result=_attempt(self.function, s.x, self.args, self.steady))
        if s.steady_result.success:
            return _Stage(self.result(True, f"Converged by the steady Newton solve after {step} "
                                      "pseudo-time steps", step, s), s)
        return _Stage(None, s)

    def _grown_dt(self, s):
        """``dt`` of the next step: grown after an easy step, by the residual reduction."""
        if s.g_prev is None or not s.history:
            return s.dt
        if s.g_norm > 0.0 and s.history[-1]["nit"] <= _EASY_NIT:     # the last step took few iterations
            return min(self.dt_max, s.dt * min(self.growth, max(_MIN_GROWTH, s.g_prev / s.g_norm)))
        return s.dt

    def _time_step(self, step, s):
        """One implicit pseudo-time step, shortening ``dt`` on failure; the result if it fails."""
        dt = self._grown_dt(s)
        x_old = s.x.ravel().copy()
        for _ in range(_STEP_TRIES):
            transient = _transient_residual(self.function, self.mass_matrix, x_old, dt)
            step_result = _attempt(transient, s.x, self.args, self.inner)
            if step_result.success:
                break
            dt *= _DT_SHRINK
        else:
            return _Stage(self.result(False, f"pseudo-time step {step} failed down to dt = {dt:.2e}", step, s), s)
        history = s.history + [{"dt": dt, "residual_norm": s.g_norm, "nit": step_result.nit}]
        if self.verbose:
            print(f"pseudo_transient step {step}: dt={dt:.2e} |g|={s.g_norm:.2e} nit={step_result.nit}")
        return _Stage(None, s._replace(x=step_result.x, dt=dt, g_prev=s.g_norm, history=history))


def _transient_residual(function, mass_matrix, x_old, dt):
    """``function`` with the implicit pseudo-time term of a step ``dt`` from ``x_old``."""
    def transient(xv, *a, _dt=dt):
        g, jac = function(xv, *a)
        g = np.asarray(g, dtype=float).ravel() + mass_matrix @ (xv.ravel() - x_old) / _dt
        return g, jac + mass_matrix / _dt
    return transient


def pseudo_transient(function, initial_guess, mass=1.0, dt=1e-3, args=(), *, dt_max=np.inf,
                     max_steps=200, growth=10.0, steady_ratio=0.1, steady_every=10,
                     newton_kwargs=None, steady_kwargs=None, verbose=0):
    """Reach a steady state of ``function(x) = 0`` by implicit pseudo-time steps.

    When Newton from the initial guess fails (ignition, extinction, a guess
    far from any solution), march ``mass dx/dt = -function(x)`` with implicit
    Euler steps instead: each step solves
    ``function(x) + mass (x - x_old) / dt = 0`` with :func:`newton`, which is
    well conditioned for small ``dt``. After a step whose Newton solve took
    at most 3 iterations ``dt`` doubles, or grows by the residual reduction
    ``||g_old|| / ||g||`` when that is larger (switched evolution relaxation;
    at most ``growth`` per step and at most ``dt_max``); after a harder step it
    is kept, and a failed step is retried with ``dt / 4``. The residual may
    rise for a while, as it does through an ignition.

    Measured on the Weisz-Hicks pellets where ``newton(...,
    damping="guarded")`` fails: the march reached the steady state in 18 of
    22, mostly within 20 steps. The four it did not reach (a steep reaction
    front crawling at tiny ``dt``) were solved by ``damping="monotone"``;
    try both. A steady Newton solve from the current state is
    tried every ``steady_every`` steps and whenever the residual norm has
    dropped by ``steady_ratio`` since the last attempt; it ends the march when
    it converges.

    Parameters
    ----------
    function : callable
        ``function(x, *args) -> (residual, jacobian)``, as for :func:`newton`,
        with a matrix Jacobian.
    initial_guess : numpy.ndarray
        Starting state.
    mass : float, numpy.ndarray or sparse matrix, optional
        Accumulation term: a scalar or an array broadcast against the state
        (a diagonal), or a matrix. It sets the relative speed of the fields in
        pseudo-time and must have the units of the residual per unit of the
        state; 1 suits residuals scaled like the state.
    dt : float, optional
        First pseudo-time step.
    args : tuple, optional
        Extra arguments of ``function``.
    dt_max : float, optional
        Largest pseudo-time step.
    max_steps : int, optional
        Largest number of accepted pseudo-time steps.
    growth : float, optional
        Largest factor by which ``dt`` grows per step.
    steady_ratio : float, optional
        Try the steady solve each time the residual norm has dropped by this
        factor.
    steady_every : int, optional
        Try the steady solve at least every so many steps.
    newton_kwargs : dict, optional
        Options for the Newton solve of each step (default ``maxfev=10``,
        ``damping="guarded"``).
    steady_kwargs : dict, optional
        Options for the steady Newton solves (default ``maxfev=30``,
        ``damping="guarded"``).
    verbose : int, optional
        ``1`` prints one line per step.

    Returns
    -------
    scipy.optimize.OptimizeResult
        ``x``, ``success``, ``message``, ``nsteps``, ``history`` (``dt``,
        residual norm and Newton iterations per step) and ``steady`` (the
        last steady :func:`newton` result).
    """
    x = np.array(initial_guess, dtype=float, order="C")
    shape = x.shape
    mass_matrix = _mass_matrix(mass, shape)
    inner = {"maxfev": 10, "damping": "guarded", **(newton_kwargs or {})}
    steady = {"maxfev": 30, "damping": "guarded", **(steady_kwargs or {})}
    run = _PseudoTransient(function=function, args=args, mass_matrix=mass_matrix, inner=inner, steady=steady,
                           dt_max=dt_max, max_steps=max_steps, growth=growth, steady_ratio=steady_ratio,
                           steady_every=steady_every, verbose=verbose)
    return run.march(x, dt)


_MAX_SECANT_RATIO = 4.0


def _continuation_predictions(path_p, path_x, p):
    """Predictions of the solution at ``p``, in the order to try them.

    Yields ``(guess, bounded)`` pairs; ``bounded`` is true for a secant that
    extrapolates at most ``_MAX_SECANT_RATIO`` times the parameter distance
    it is based on. First the secant through the last two solutions. When
    its ratio exceeds the bound it amplifies the Newton tolerance error of
    those two solutions: nearly equal parameter values (``0.1`` and
    ``1 - 0.9``) give a ratio near 1e14 and a prediction no halving repairs.
    It is then followed by the secant from the most recent earlier solution
    within the bound, or by the last solution when there is none.
    """
    if len(path_p) < 2:
        yield path_x[-1].copy(), False
        return
    step = p - path_p[-1]
    base = path_p[-1] - path_p[-2]
    if base != 0:
        ratio = step / base
        yield path_x[-1] + ratio * (path_x[-1] - path_x[-2]), abs(ratio) <= _MAX_SECANT_RATIO
        if abs(ratio) <= _MAX_SECANT_RATIO:
            return
    guess = _earlier_secant(path_p, path_x, step)
    yield (path_x[-1].copy(), False) if guess is None else (guess, True)


def _earlier_secant(path_p, path_x, step):
    """The secant over ``step`` from the most recent solution before the last two that lies within the
    bound, None when there is none."""
    for k in range(len(path_p) - 3, -1, -1):
        base = path_p[-1] - path_p[k]
        if base != 0 and abs(step) <= _MAX_SECANT_RATIO * abs(base):
            return path_x[-1] + (step / base) * (path_x[-1] - path_x[k])
    return None


class _Correction(NamedTuple):
    """One corrector attempt at a sub-step: the newton result (None below round-off), whether it
    jumped to another branch, and whether its prediction was a secant within the bound."""

    res: object
    jumped: bool
    bounded: bool

    @property
    def ok(self):
        """Whether the sub-step is accepted: converged without a jump."""
        return self.res is not None and self.res.success and not self.jumped


class _ContinuationSettings(NamedTuple):
    """The options of one :func:`continuation` run: newton's, and its own."""

    options: dict
    max_change: Optional[float]
    max_halvings: int
    verbose: object


class _Continuation:
    """One :func:`continuation` run: the path of solutions and those at the requested values."""

    def __init__(self, function, settings, p, first):
        """Start from the solution ``first`` at the first value ``p``."""
        self.function, self.settings = function, settings
        self.done, self.xs, self.results = [p], [first.x.copy()], [first]
        self.path_p, self.path_x = [p], [first.x.copy()]

    def repeat(self, target):
        """A repeated value: nothing to solve."""
        self.done.append(target)
        self.xs.append(self.path_x[-1].copy())
        self.results.append(self.results[-1])

    def _jumped(self, res, guess):
        """Whether the converged ``res`` moved more than ``max_change`` from its prediction."""
        if not res.success or self.settings.max_change is None:
            return False
        # The first step's only prediction is the last solution itself (no secant yet): the test starts at the second.
        if len(self.path_p) < 2:
            return False
        size = max(float(np.max(np.abs(self.path_x[-1]))), float(np.max(np.abs(res.x))), np.finfo(float).tiny)
        return float(np.max(np.abs(res.x - guess))) > self.settings.max_change * size

    def correct(self, p):
        """Newton at ``p`` from each prediction in turn: the last :class:`_Correction`."""
        correction = _Correction(None, False, False)
        if p == self.path_p[-1]:            # the sub-step is below round-off
            return correction
        for guess, bounded in _continuation_predictions(self.path_p, self.path_x, p):
            res = _attempt(lambda xv, _p=p: self.function(xv, _p), guess, (), self.settings.options)
            correction = _Correction(res, self._jumped(res, guess), bounded)
            if correction.ok:
                break
        return correction

    def _accept(self, p, res, target):
        """Add the solution at ``p`` to the path; whether it is the one at ``target``."""
        self.path_p.append(p)
        self.path_x.append(res.x.copy())
        if self.settings.verbose:
            print(f"continuation p={p:g} nit={res.nit}")
        if p == target:
            self.done.append(target)
            self.xs.append(res.x.copy())
            self.results.append(res)
            return True
        return False

    def _halved(self, level, step, tried):
        """The next sub-step level, at least one halving and short of ``tried``."""
        level += 1
        while level <= self.settings.max_halvings and abs(step) * 0.5**level >= tried:
            level += 1
        return level

    def reach(self, target):
        """Sub-step along the path to ``target``; the failure result, None on success."""
        # Sub-steps are the requested step divided by 2**level. A failure
        # halves the sub-step; an accepted one doubles the next (at most to
        # the requested step), so a stretch needing short sub-steps does not
        # use up the halving budget while every sub-step succeeds.
        path_p, max_halvings = self.path_p, self.settings.max_halvings
        step = target - path_p[-1]
        level = 0
        p = target
        while True:
            correction = self.correct(p)
            if not correction.ok:
                # Halve until the sub-step is shorter than the one that failed: a
                # failure at the capped last sub-step would otherwise be retried
                # with the same prediction.
                level = self._halved(level, step, abs(p - path_p[-1]))
                if level > max_halvings or correction.res is None:
                    return self._stopped(target, correction)
                p = path_p[-1] + step * 0.5**level
                continue
            if self._accept(p, correction.res, target):
                return None
            level = max(level - 1, 0)
            p = _capped(p, step * 0.5**level, target)

    def march(self, targets):
        """Reach each of ``targets`` in turn: the result of :func:`continuation`."""
        for target in targets:
            if target == self.path_p[-1]:            # a repeated value: nothing to solve
                self.repeat(target)
            elif (failed := self.reach(target)) is not None:
                return failed
        return OptimizeResult(x=self.path_x[-1], success=True, params=self.done, xs=self.xs, results=self.results,
                              last_param=self.path_p[-1], message="Reached every parameter value")

    def _stopped(self, target, correction):
        """The failure result of a march stopped on the way to ``target``."""
        why = _continuation_failure(correction.res, correction.jumped, correction.bounded)
        return OptimizeResult(
            x=self.path_x[-1], success=False, params=self.done, xs=self.xs,
            results=self.results, last_param=self.path_p[-1],
            message=f"stopped at p = {self.path_p[-1]:g} on the way to {target:g}: {why}")


def _capped(p, h, target):
    """The next sub-step value ``p + h``, or ``target`` when it is at most ``h`` away."""
    return target if abs(target - p) <= abs(h) else p + h


def _failure_cause(res, jumped, bounded):
    """The cause in :func:`_continuation_failure`, before the predictor suffix."""
    if res is None:
        return "the sub-step fell below round-off"
    if jumped:
        why = "the corrector jumps to another solution branch"
        if bounded:
            why += "; probably a turning point (fold) of the solution branch"
        return why
    if res.status == _Status.NON_FINITE and res.nit == 0:
        return (f"the residual cannot be evaluated at the prediction ({res.message}); "
                "it lies outside the residual's domain")
    why = f"the corrector fails ({res.message})"
    if bounded:
        why += ("; a turning point (fold) of the solution branch, or a "
                "corrector that cannot converge here (an inexact Jacobian, "
                "such as NumJac's eps_jac, or a tolerance below round-off)")
    return why


def _continuation_failure(res, jumped, bounded):
    """Why continuation stopped after the sub-step that gave ``res``."""
    why = _failure_cause(res, jumped, bounded)
    if res is not None and not bounded:
        why += (" from the constant predictor (no earlier solution far enough "
                "back for a secant)")
    return why


def continuation(function, initial_guess, params, *, max_halvings=6, max_change=0.2,
                 newton_kwargs=None, verbose=0):
    """Solve ``function(x, p) = 0`` along a sequence of parameter values.

    Natural-parameter continuation: each point starts from a secant
    extrapolation of the two previous solutions and is corrected with
    :func:`newton`. A failed correction halves the parameter step; an
    accepted sub-step doubles the next one again, up to the requested step.
    The march stops when a sub-step would have to be shorter than the
    requested step divided by ``2**max_halvings``. A prediction at which
    ``function`` raises one of newton's ``recoverable`` exceptions counts as
    a failed correction. A correction that moves
    ``x`` by more than ``max_change`` times ``max |x|`` from the prediction
    counts as failed: near a turning point (fold) the corrector otherwise
    jumps silently to another solution branch. The march then stops and
    reports where; the fold lies between the last parameter reached and the
    next one. Following a branch around a fold needs arclength continuation,
    which this function does not do.

    A secant over more than 4 times the step it is based on also multiplies
    the Newton tolerance error of the two solutions, without bound for nearly
    equal parameter values (``0.1`` and ``1 - 0.9``). When such a prediction
    fails, the step is retried before any halving from the secant through an
    earlier solution at least a quarter of the step back, or else from the
    last solution. Nearly equal values in ``params`` are therefore harmless;
    exactly equal ones are not solved again. A fold is offered in the
    message only when the last prediction tried was a secant within that
    bound, and when the corrector failed (rather than jumped) it is named
    beside the other cause: a corrector that cannot converge at that point,
    for instance with an inexact Jacobian.

    Parameters
    ----------
    function : callable
        ``function(x, p) -> (residual, jacobian)``.
    initial_guess : numpy.ndarray
        Guess for the first parameter value.
    params : sequence of float
        Parameter values, in the order to visit them.
    max_halvings : int, optional
        Largest number of consecutive halvings of the requested step: the
        shortest sub-step is the requested step divided by ``2**max_halvings``.
    max_change : float or None, optional
        Largest accepted relative distance between prediction and solution;
        ``None`` disables the branch-jump check.
    newton_kwargs : dict, optional
        Options for :func:`newton`.
    verbose : int, optional
        ``1`` prints one line per accepted point.

    Returns
    -------
    scipy.optimize.OptimizeResult
        ``success`` (every value reached), ``message``, ``params`` (the values
        reached), ``xs`` (their solutions), ``results`` (their :func:`newton`
        results), ``last_param`` (the last parameter at which a solution was
        found, possibly an intermediate one) and ``x`` (that solution).
    """
    params = [float(p) for p in params]
    options = dict(newton_kwargs or {})
    first = _attempt(lambda xv: function(xv, params[0]), np.array(initial_guess, dtype=float),
                     (), options)
    if not first.success:
        return OptimizeResult(x=first.x, success=False, params=[], xs=[], results=[first],
                              last_param=None,
                              message=f"no solution at the first value {params[0]:g}: "
                                      f"{first.message}")
    settings = _ContinuationSettings(options=options, max_change=max_change, max_halvings=max_halvings, verbose=verbose)
    run = _Continuation(function, settings, params[0], first)
    return run.march(params[1:])


def clip_approach(values, dummy, lower_bounds=0, upper_bounds=None, factor=0):
    """Project values onto bounds, optionally with a relaxed approach rule.

    Parameters
    ----------
    values : numpy.ndarray
        Values to modify in place.
    dummy : Any
        Placeholder argument kept for API compatibility.
    lower_bounds, upper_bounds : float or numpy.ndarray, optional
        Lower and upper bounds. Scalars and broadcastable arrays are supported.
    factor : float, optional
        Relaxation factor for out-of-bound entries. ``0`` applies strict clipping.
        Non-zero values apply a linear approach update toward the violated bound.
    """
    if factor == 0:
        np.clip(values, lower_bounds, upper_bounds, out=values)
    else:
        if lower_bounds is not None:
            _approach_bound(values, values < lower_bounds, lower_bounds, factor)
        if upper_bounds is not None:
            _approach_bound(values, values > upper_bounds, upper_bounds, factor)


def _approach_bound(values, mask, bounds, factor):
    """In place: the ``mask`` entries of ``values`` past ``bounds``, moved by the approach rule."""
    if np.any(mask):
        b = np.broadcast_to(bounds, values.shape)
        values[mask] = (1.0 + factor) * b[mask] - factor * values[mask]
