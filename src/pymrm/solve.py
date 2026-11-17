"""Nonlinear-solver utilities used by :mod:`pymrm`."""

import warnings
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from scipy import sparse
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve, solve as dense_solve
from scipy.optimize import Bounds, OptimizeResult
from scipy.sparse import linalg

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
_UNTESTED = float("nan")  # theta of a step no monotonicity test measured
_BACKWARD_FLOOR = 1.0  # weight of the field scale in the backward error
_GLOBAL_FLOOR = 1.0e-6  # smallest field scale, relative to the largest unknown

# Exceptions that mark an evaluation point as unusable (``newton``'s default).
_RECOVERABLE = (FloatingPointError, OverflowError, ZeroDivisionError)

_MESSAGES = {
    0: "Converged",
    1: "Converged to the stagnation floor",
    2: "Did not converge",
    3: "Damping fell below its minimum",
    4: "Non-finite residual",
    5: "Singular Jacobian",
    6: "Linear solver failed",
    7: "Step criterion met but the backward error is above backward_tol",
    8: "Stopped on a bound",
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
    if solver is None or (isinstance(solver, str) and solver in ("spsolve", "lu")):
        return "direct"
    if isinstance(solver, str) and solver in ("cg", "bicgstab"):
        return solver
    if isinstance(solver, str) and solver == "splu":
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

    def __init__(self, solver, kwargs, row_scaling, legacy=False):
        self.kwargs = dict(kwargs or {})
        self.row_scaling = row_scaling
        # legacy: reproduce spsolve and scipy.linalg.solve of pymrm 2.5.0 bit for bit
        self.legacy = legacy and not row_scaling
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
            # scipy.linalg.solve detects structure (symmetric, triangular), as before
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LinAlgWarning)
            factor = lu_factor(a, check_finite=False)
        if (factor[0].diagonal() == 0.0).any():
            raise _SingularJacobian("exactly singular")
        if r is None:
            return (lambda rhs: lu_solve(factor, rhs, check_finite=False)), None
        return (lambda rhs: lu_solve(factor, r * rhs, check_finite=False)), r

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

    @property
    def active(self):
        return self.lower is not None or self.upper is not None

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
        (Remembering every component ever held, not only the last step's, lost
        two of 108 bounded Arrhenius pellets.)
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
        """Record the components :meth:`project` held in the accepted ``trial``."""
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
    largest = float(np.max(ax)) if x.size else 0.0
    # The initial guess carries the problem's scale when the solution is ~0.
    floor = _GLOBAL_FLOOR * max(largest, x0_size)
    if len(shape) >= 2:
        # The floor also keeps a single-field state stored without a field
        # axis, (nz, nr), from being judged column by column; a trace field is
        # still judged far below its neighbours.
        ref = np.maximum(np.max(ax.reshape(-1, shape[-1]), axis=0), floor)
        return np.broadcast_to(ref, shape).reshape(-1)
    return np.full(x.size, max(largest, floor))


def _backward_error(g, jac, x, x_ref):
    """Componentwise backward error with a floor relative to the field scale.

    ``max_i |g_i| / (|J| (|x| + eta x_ref) + |J x - g|)_i`` with
    ``eta = _BACKWARD_FLOOR``: the measure of
    :func:`pymrm.checks.residual_check`, except that components far below the
    scale of their field (a reaction dead zone, an absent species) are judged
    against that scale instead of against their own size.
    """
    if sparse.issparse(jac):
        abs_jac = abs(jac)
    else:
        jac = np.asarray(jac, dtype=float)
        abs_jac = np.abs(jac)
    denominator = abs_jac @ (np.abs(x) + _BACKWARD_FLOOR * x_ref) + np.abs(jac @ x - g)
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
    d_user = None if scale is None else \
        np.array(np.broadcast_to(np.asarray(scale, dtype=float), shape)).ravel()
    if d_user is not None and not (np.all(np.isfinite(d_user)) and np.all(d_user > 0.0)):
        raise ValueError("scale must be finite and positive for every unknown.")
    return d_user


class _Eval(NamedTuple):
    """One evaluation: the residual as returned, flat, and the Jacobian."""

    g_raw: object
    g: np.ndarray
    jac: object


class _Newton(NamedTuple):
    """The factorised Jacobian and the Newton step it gives."""

    lin_solve: object
    r: object
    dx: np.ndarray


class _Step(NamedTuple):
    """One iteration's Newton step and its tests, read-only for the phases of the iteration."""

    ev: _Eval
    lin_solve: object
    r: object
    dx: np.ndarray
    dy: np.ndarray
    step_inf: float
    converged: bool
    near: bool
    gate: bool
    stagnant: bool
    be: float
    last_it: bool
    last: bool


class _Carry(NamedTuple):
    """What an iteration hands to the next, to the verbose line and to ``_exhausted``."""

    ev: _Eval
    step_inf: float
    be: float
    gate_failed: bool
    clipped: int
    lam: float
    theta: float
    lam_next: float


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


class _Best(NamedTuple):
    """The Armijo trial with the smallest scaled residual ``phi`` so far."""

    phi: float
    trial: _Trial


class _SearchCtx(NamedTuple):
    """What the tests of one damped search compare with: the Newton step and the residual at x."""

    dy: np.ndarray
    dy_rms: float
    lin_solve: object
    row: object
    phi0: float


class _Verdict(NamedTuple):
    """A test's verdict on a trial: the accepted trial (or None), the next lam, theta and the best trial."""

    accepted: _Trial
    lam: float
    theta: float
    best: _Best


class _SearchEnd(NamedTuple):
    """How a damped search ended: the accepted trial (or None), the trial at lam_max and the last theta."""

    accepted: _Trial
    at_lam_max: _Trial
    theta: float


class _Found(NamedTuple):
    """The step of a damped iteration (``trial.ev`` None: the damping failed), its theta and the next lam."""

    trial: _Trial
    theta: float
    lam_next: float


def _lam_min(near):
    """The damping below which the search ends."""
    # Near the solution a failed test is usually round-off: search briefly.
    return _LAM_NEAR if near else _LAM_MIN


def _search_context(s):
    """The ``_SearchCtx`` of the iteration ``s``."""
    dy_rms = _rms(s.dy)
    r, jac = s.r, s.ev.jac
    row = r if r is not None else (_row_scale(jac) if _is_matrix(jac) else 1.0)
    phi0 = float(np.linalg.norm(row * s.ev.g))
    return _SearchCtx(s.dy, dy_rms, s.lin_solve, row, phi0)


def _stop_status(s, converged):
    """0 if ``converged``, 1 if stalled, None to iterate on (both need the backward-error gate of ``s``)."""
    if not s.gate:
        return None
    if converged:
        return 0
    if s.stagnant:
        return 1
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


@dataclass(eq=False, slots=True)
class _NewtonRun:
    """One :func:`newton` solve: the problem, its settings and the iteration counters."""

    function: object
    args: tuple
    recoverable: tuple
    x: np.ndarray
    x_flat: np.ndarray
    shape: tuple
    n: int
    x0_size: float
    d_user: object
    d: np.ndarray
    lin: _LinearSolver
    mode: str
    legacy_exact: bool
    relative: bool
    tol: object
    rtol: float
    tol_vec: object
    bnd: _Bounds
    use_stagnation: bool
    maxfev: object
    callback: object
    refresh: object
    backward_tol: object
    verbose: object
    nfev: int = 0
    nfact: int = 0
    history: list = field(default_factory=list)

    def _call(self, xv, guard, last):
        """``function(xv, *args)`` as ``(g_raw, jac)``; None if guarded and recoverable."""
        if not guard:
            g_raw, jac = self.function(xv, *self.args)
            return g_raw, jac
        try:
            g_raw, jac = self.function(xv, *self.args)
        except self.recoverable:
            return None
        except Exception as exc:
            if last:
                # The old iteration never evaluated here: report, do not raise.
                raise _LastEvaluationFailed(f"{type(exc).__name__}: {exc}") from exc
            raise
        return g_raw, jac

    def _evaluate(self, xv, guard, last):
        """Residual (flat) and Jacobian at ``xv``, finite or not; None if guarded and recoverable."""
        self.nfev += 1
        out = self._call(xv, guard, last)
        if out is None:
            return None
        g_raw, jac = out
        g_raw = g_raw.toarray() if sparse.issparse(g_raw) else np.array(g_raw, dtype=float)
        g_flat = g_raw.reshape(-1)
        if sparse.issparse(jac) and jac.format not in ("csr", "csc", "coo", "bsr", "dia"):
            jac = jac.tocsc()      # lil, dok: no .data array of the stored values
        return _Eval(g_raw, g_flat, jac)

    def evaluate(self, xv, guard, last=False):
        """Residual (flat) and Jacobian at ``xv``; None if unusable and guarded."""
        ev = self._evaluate(xv, guard, last)
        if ev is None or not _is_finite(ev):
            return None
        return ev

    def backward_error(self, g_flat, jac):
        if not _is_matrix(jac):
            return float("nan")
        return _backward_error(g_flat, jac, self.x_flat,
                               _field_reference(self.x_flat, self.shape, self.d_user, self.x0_size))

    def _outcome(self, status, end, extra):
        """The result of the run ending with ``status`` at ``end``."""
        ev = end.ev
        g_raw = None if ev is None else ev.g_raw
        message = _MESSAGES[status] + (f": {extra}" if extra else "")
        return OptimizeResult(x=self.x, success=status in (0, 1), status=status, message=message,
                              nit=len(self.history), nfev=self.nfev, nfact=self.nfact, fun=g_raw,
                              step_norm=end.step_norm, backward_error=end.be, history=self.history)

    def finish(self, status, end, extra=""):
        """The result of the run ending at ``end``, raised in a ``breakdown`` for the statuses 4, 5 and 6."""
        result = self._outcome(status, end, extra)
        if status in (4, 5, 6):
            # No usable Newton step: raise, as these cases did up to 2.5.0
            # (callers catch ValueError, LinAlgError or RuntimeError).
            raise NewtonBreakdown(result.message, result)
        return result

    def solve(self, guard_guess):
        """Evaluate at the initial guess and iterate; the result of :func:`newton`."""
        if guard_guess:
            # The guess is a prediction (continuation, pseudo_transient): one of
            # ``recoverable`` there marks it unusable, like a non-finite residual.
            try:
                ev = self._evaluate(self.x, guard=False, last=False)
            except self.recoverable as exc:
                return self.finish(4, _End(None, float("nan"), float("nan")),
                                   f"{type(exc).__name__} at the initial guess ({exc})")
        else:
            ev = self._evaluate(self.x, guard=False, last=False)
        if not _is_finite(ev):
            return self.finish(4, _End(ev, float("nan"), float("nan")), "at the initial guess")
        return self.iterate(ev)

    def _newton_step(self, ev, step_inf):
        """Factorise the Jacobian at ``ev`` and solve for the Newton step: ``(solve, r, dx)``.

        A singular Jacobian, a failed linear solve or a non-finite step ends the
        run: ``finish`` raises for the statuses 5 and 6.
        """
        g_raw, g, jac = ev
        try:
            solve, r = self.lin.factorize(jac)
            self.nfact += 1
            dx = np.asarray(solve(g_raw if self.lin.kind == "callable" else g),
                            dtype=float).reshape(-1)
        except _SingularJacobian as exc:
            self.finish(5, _End(ev, step_inf, float("nan")), str(exc))     # raises
        except _LinearSolverFailed as exc:
            result = self._outcome(6, _End(ev, step_inf, float("nan")), str(exc))
            raise KrylovBreakdown(result.message, result)
        if not np.isfinite(dx).all():
            self.finish(6, _End(ev, step_inf, float("nan")), "non-finite Newton step")     # raises
        return _Newton(solve, r, dx)

    def iterate(self, ev):
        carry = _Carry(ev, float("nan"), float("nan"), False, 0, 1.0, _UNTESTED, 1.0)
        window = []          # step norms of the recent full steps at the floor
        maxfev = int(self.maxfev)
        for it in range(maxfev):
            s = self._measure(carry.ev, self._newton_step(carry.ev, carry.step_inf), it == maxfev - 1, window)
            out = self._plain_step(s) if self.mode == "none" else self._damped_step(s, carry, it)
            if isinstance(out, OptimizeResult):
                return out
            carry = out
            # Report the iteration and update the stagnation window.
            if self.verbose:
                print(f"newton it={it:3d} |dx|={carry.step_inf:.3e} lam={carry.lam:.3g} "
                      f"theta={carry.theta:.3g} clipped={carry.clipped} nfev={self.nfev}")
            _slide(window, s, self.use_stagnation)
        return self._exhausted(carry)

    def _step_test(self, dy, step_inf):
        """Whether the step ``dy`` (scaled) meets the tolerance, and by how much it exceeds it."""
        tol = self.tol
        y = self.x_flat / self.d
        if self.relative:
            limit = self.tol_vec + self.rtol * np.abs(y)
            converged = bool(np.all(np.abs(dy) <= limit))
            with np.errstate(divide="ignore", invalid="ignore"):
                excess = float(np.max(np.where(dy == 0.0, 0.0, np.abs(dy) / limit)))
        else:
            converged = step_inf < tol
            excess = step_inf / tol if tol > 0 else (0.0 if step_inf == 0 else np.inf)
        if not self.legacy_exact:
            converged = converged or step_inf == 0.0
        return converged, excess

    def _measure(self, ev, newton, last_it, window):
        """Step norm, convergence and stagnation tests and backward error of this iteration."""
        dy = newton.dx / self.d
        step_inf = float(np.abs(dy).max()) if self.n else 0.0
        converged, excess = self._step_test(dy, step_inf)
        near = converged or (self.refresh is None and excess < _NEAR)
        last = converged or last_it

        # The verdict uses the residual at x, before the step: no evaluation
        # beyond those of plain Newton. Near a solution the step only improves it.
        be = float("nan")
        measured = near or last_it or self.relative
        if measured:
            be = self.backward_error(ev.g, ev.jac)
        backward_tol = self.backward_tol
        gate = backward_tol is None or np.isnan(be) or be <= backward_tol
        stagnant = self.use_stagnation and near and _stalled(window, step_inf)
        return _Step(ev, newton.lin_solve, newton.r, newton.dx, dy, step_inf, converged, near, gate, stagnant, be,
                     last_it, last)

    def _plain_step(self, s):
        """Plain (or bound-shortened) step; callback and refresh before the
        next evaluation, as the callers of the undamped iteration expect."""
        x_flat, dx, bnd = self.x_flat, s.dx, self.bnd
        lam = 1.0
        if not s.near and bnd.active:
            lam = max(_LAM_FLOOR, bnd.max_step(x_flat, dx))
        t = self._project(s, lam)
        self._take(s, t, _UNTESTED)
        return self._plain_verdict(s, t)

    def _plain_verdict(self, s, t):
        """Stop after a plain step, or evaluate at the new ``x``: the result or the ``_Carry``."""
        converged, step_inf, be = s.converged, s.step_inf, s.be
        if self.relative and step_inf > 0.0:
            # The relative test uses x after the step (and the callback).
            converged = bool(np.all(np.abs(s.dy) <= self.tol_vec + self.rtol * np.abs(self.x_flat / self.d)))
        status = _stop_status(s, converged)
        if status is not None:
            return self._final(s, status, t.clipped)
        if s.last_it:
            # plain Newton does not evaluate after its last step
            return self._exhausted(_Carry(s.ev, step_inf, be, converged, t.clipped, t.lam, _UNTESTED, 1.0))
        ev_new = self.evaluate(self.x, guard=False)
        if ev_new is None:
            return self.finish(4, _End(s.ev, step_inf, float("nan")), "after a full step")
        return _Carry(ev_new, step_inf, be, converged, t.clipped, t.lam, _UNTESTED, 1.0)

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
        if status == 0:
            return self.finish(0, _End(s.ev, step_inf, s.be))
        if clipped:
            return self.finish(8, _End(s.ev, step_inf, s.be), f"{clipped} components held by a bound")
        return self.finish(1, _End(s.ev, step_inf, s.be), f"|dx| = {step_inf:.2e} (tol {np.max(self.tol):.1e})")

    def _final_step(self, s, status):
        """Take the converged or stalled step without evaluating there, as plain Newton did."""
        t = self._project(s, 1.0)
        self._take(s, t, float("nan"))
        return self._final(s, status, t.clipped)

    def _damped_step(self, s, carry, it):
        """Stop, or find a damped step, accept it and evaluate there: the result or the ``_Carry``."""
        status = _stop_status(s, s.converged)
        if status is not None:
            # _final_step takes the stalled step and reports 8 only if that step is held
            return self._final_step(s, status)
        try:
            found = self._find_step(s, carry.lam_next)
            if found.trial.ev is None:
                return self._damping_failed(s, found.trial.clipped, it)
            return self._accept(s, found)
        except _LastEvaluationFailed as exc:
            return self.finish(2, _End(s.ev, s.step_inf, float("nan")),
                               f"the residual at the last trial point raised {exc}")

    def _damping_failed(self, s, clipped, it):
        """The result when no damped step was found (status 3)."""
        extra = f"{_lam_min(s.near):g} at iteration {it}"
        if clipped:
            extra += f"; {clipped} components held by a bound"
        return self.finish(3, _End(s.ev, s.step_inf, self.backward_error(s.ev.g, s.ev.jac)), extra)

    def _project(self, s, lam):
        """The trial point ``x - lam dx``, held inside the bounds, not evaluated."""
        x_flat, dx, bnd = self.x_flat, s.dx, self.bnd
        trial = x_flat - lam * dx
        clipped = bnd.project(trial, x_flat, dx) if bnd.active else 0
        return _Trial(trial, None, lam, clipped)

    def _trial(self, s, lam):
        """The trial point ``x - lam dx`` (held inside the bounds) and its evaluation."""
        t = self._project(s, lam)
        return _Trial(t.x, self.evaluate(t.x.reshape(self.shape), guard=True, last=s.last), lam, t.clipped)

    def _find_step(self, s, lam_next):
        """The ``_Found`` step of this iteration, the damped search starting from ``lam_next``."""
        bnd = self.bnd
        lam_max = max(_LAM_FLOOR, bnd.max_step(self.x_flat, s.dx)) if bnd.active else 1.0
        at_lam_max = None   # the trial at lam_max, if made
        if self.mode == "guarded":
            # Full step (shortened only by the bounds far from the solution).
            lam = 1.0 if s.near else lam_max
            t = self._trial(s, lam)
            if lam == lam_max:
                at_lam_max = t
            if t.ev is not None:
                return _Found(t, _UNTESTED, lam_next)
            lam_next = 0.25 * lam
        return self._line_search(s, min(lam_next, lam_max), lam_max, at_lam_max)

    def _line_search(self, s, lam, lam_max, at_lam_max):
        """The damped search from ``lam``, else the escape: the ``_Found`` step and the next lam."""
        end = self._search(s, lam, lam_max, at_lam_max)
        if end.accepted is not None:
            lam_next = 1.0 if self.mode == "guarded" else min(1.0, 4.0 * end.accepted.lam)
            return _Found(end.accepted, end.theta, lam_next)
        # lam_next matters only if the escape succeeds
        return _Found(self._escape(s, lam_max, end.at_lam_max), end.theta, 1.0)

    def _search(self, s, lam, lam_max, at_lam_max):
        """Damped search: monotonicity test or Armijo, from ``lam``; the best Armijo trial if none passes."""
        ctx = _search_context(s)
        lam_min = _lam_min(s.near)
        v = _Verdict(None, lam, _UNTESTED, None)
        while v.lam >= lam_min:
            t = self._trial(s, v.lam)
            v = self._judge(t, ctx, v)
            if t.lam == lam_max:
                at_lam_max = t
            if v.accepted is not None:
                return _SearchEnd(v.accepted, at_lam_max, v.theta)
        return _SearchEnd(self._restore_best(v.best, ctx.phi0, s.last), at_lam_max, v.theta)

    def _judge(self, t, ctx, v):
        """The ``_Verdict`` on the trial ``t``: a quarter lam if it is unusable, else the mode's test."""
        if t.ev is None:
            return _Verdict(None, v.lam * 0.25, v.theta, v.best)
        if self.mode == "armijo":
            return self._armijo_test(t, ctx, v)
        return self._monotonicity_test(t, ctx, v)

    def _armijo_test(self, t, ctx, v):
        """The trial if it decreases the scaled residual enough, else a halved lam (and the best trial)."""
        lam = t.lam
        phi = float(np.linalg.norm(ctx.row * t.ev.g))
        if phi <= (1.0 - _ARMIJO_C * lam) * ctx.phi0:
            return _Verdict(t, lam, v.theta, v.best)
        best = v.best
        if best is None or phi < best.phi:
            best = _Best(phi, t)
        lam *= 0.5
        return _Verdict(None, lam, v.theta, best)

    def _monotonicity_test(self, t, ctx, v):
        """The trial if its simplified Newton step contracts, else a smaller lam."""
        dy, dy_rms, lam = ctx.dy, ctx.dy_rms, t.lam
        try:
            rhs = t.ev.g_raw if self.lin.kind == "callable" else t.ev.g
            dbar = np.asarray(ctx.lin_solve(rhs), dtype=float).reshape(-1) / self.d
        except _LinearSolverFailed:
            dbar = np.full(self.n, np.inf)    # no test possible: reject the trial
        theta = _rms(dbar) / dy_rms if dy_rms > 0.0 else 0.0
        if not np.isfinite(theta):
            lam *= 0.5
            return _Verdict(None, lam, theta, v.best)
        if theta <= 1.0 - lam / 4.0:
            return _Verdict(t, lam, theta, v.best)
        mu = 0.5 * dy_rms * lam**2 / max(_rms(dbar - (1.0 - lam) * dy), 1.0e-300)
        lam = min(0.5 * lam, max(mu, 0.1 * lam))
        return _Verdict(None, lam, theta, v.best)

    def _restore_best(self, best, phi0, last):
        """The best Armijo trial if it decreased the residual, re-evaluated; else None."""
        if best is None or not best.phi < phi0:
            return None
        t = best.trial
        ev_t = self.evaluate(t.x.reshape(self.shape), guard=True, last=last)  # restore model state
        if ev_t is not None:
            return _Trial(t.x, ev_t, t.lam, t.clipped)
        return None

    def _escape(self, s, lam_max, at_lam_max):
        """The (bounded) full step, its ``ev`` None unless its residual is finite."""
        if at_lam_max is None:
            return self._trial(s, lam_max)
        t = at_lam_max
        if t.ev is not None:
            t = _Trial(t.x, self.evaluate(t.x.reshape(self.shape), guard=True, last=s.last), lam_max, t.clipped)
        return t

    def _accept(self, s, found):
        """Take the found damped step, then the callback and refresh: the ``_Carry`` or the result."""
        t = found.trial
        ev = t.ev
        if self._take(s, t, found.theta):
            ev = self.evaluate(self.x, guard=True, last=s.last)
            if ev is None:
                return self.finish(4, _End(t.ev, s.step_inf, float("nan")),
                                   "after the callback or refresh")
        return _Carry(ev, s.step_inf, s.be, s.converged, t.clipped, t.lam, found.theta, found.lam_next)

    def _exhausted(self, carry):
        """The result after ``maxfev`` iterations without convergence."""
        ev, step_inf, be = carry.ev, carry.step_inf, carry.be
        if self.mode != "none":
            be = self.backward_error(ev.g, ev.jac)     # ev is at x after a damped step
        if carry.gate_failed:
            return self.finish(7, _End(ev, step_inf, be), f"{be:.2e} after maxfev iterations")
        if carry.clipped:
            return self.finish(8, _End(ev, step_inf, be), f"{carry.clipped} components held by a bound")
        return self.finish(2, _End(ev, step_inf, be))


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

    if lin_solver_kwargs is None:
        lin_solver_kwargs = {}
    lin = _LinearSolver(solver, lin_solver_kwargs, row_scaling)

    mode = _damping_mode(damping, callback, lin)
    lin.legacy = mode == "none" and not row_scaling
    legacy_exact = mode == "none" and backward_tol is None

    d_user = _scale_vector(scale, shape)
    d = np.ones(n) if d_user is None else d_user
    relative = rtol > 0 or not np.isscalar(tol)
    tol_vec = np.array(np.broadcast_to(np.asarray(tol, dtype=float), shape)).ravel() \
        if relative else None
    bnd = _Bounds(*_parse_bounds(bounds, shape), tau)
    if bnd.active:
        bnd.enforce_initial(x_flat)
    use_stagnation = stagnation and refresh is None

    run = _NewtonRun(function=function, args=args, recoverable=recoverable, x=x, x_flat=x_flat,
                     shape=shape, n=n, x0_size=x0_size, d_user=d_user, d=d, lin=lin, mode=mode,
                     legacy_exact=legacy_exact, relative=relative, tol=tol, rtol=rtol,
                     tol_vec=tol_vec, bnd=bnd, use_stagnation=use_stagnation, maxfev=maxfev,
                     callback=callback, refresh=refresh, backward_tol=backward_tol, verbose=verbose)
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


@dataclass(eq=False)
class _PseudoTransient:
    """One :func:`pseudo_transient` march: settings, state and the steady attempts."""

    function: object
    args: tuple
    x: np.ndarray
    mass_matrix: object
    inner: dict
    steady: dict
    dt: float
    dt_max: float
    max_steps: object
    growth: float
    steady_ratio: float
    steady_every: object
    verbose: object
    history: list = field(default_factory=list)
    steady_result: object = None
    g_prev: object = None
    g_attempt: object = None

    def result(self, success, message, nsteps):
        steady_result = self.steady_result
        xr = steady_result.x if (success and steady_result is not None) else self.x
        return OptimizeResult(x=xr, success=success, message=message, nsteps=nsteps,
                              history=self.history, steady=steady_result)

    def march(self):
        """Pseudo-time steps with steady Newton attempts; the result of :func:`pseudo_transient`."""
        for step in range(int(self.max_steps) + 1):
            out = self._steady_check(step)
            if out is not None:
                return out
            if step == self.max_steps:
                break
            out = self._time_step(step)
            if out is not None:
                return out
        return self.result(False, "max_steps reached without a converged steady solve", int(self.max_steps))

    def _steady_due(self, step, g_norm):
        """Whether to try the steady Newton solve at this step."""
        if self.g_attempt is None or g_norm <= self.steady_ratio * self.g_attempt:
            return True
        return step == self.max_steps or (self.steady_every and step % self.steady_every == 0)

    def _steady_check(self, step):
        """Residual norm at ``x`` and, when due, a steady solve; the result if the march ends."""
        g_now, _ = self.function(self.x, *self.args)
        self.g_norm = g_norm = float(np.linalg.norm(np.asarray(g_now, dtype=float).ravel()))
        if not np.isfinite(g_norm):
            return self.result(False, "non-finite residual during the pseudo-time march", step)
        if self._steady_due(step, g_norm):
            self.g_attempt = g_norm
            self.steady_result = _attempt(self.function, self.x, self.args, self.steady)
            if self.steady_result.success:
                return self.result(True, f"Converged by the steady Newton solve after {step} "
                                   "pseudo-time steps", step)
        return None

    def _time_step(self, step):
        """One implicit pseudo-time step, shortening ``dt`` on failure; the result if it fails."""
        g_norm, g_prev, history = self.g_norm, self.g_prev, self.history
        easy = bool(history) and history[-1]["nit"] <= 3     # the last step took few iterations
        if g_prev is not None and g_norm > 0.0 and easy:
            self.dt = min(self.dt_max, self.dt * min(self.growth, max(2.0, g_prev / g_norm)))
        self.g_prev = g_norm
        x_old = self.x.ravel().copy()
        for _ in range(20):
            transient = _transient_residual(self.function, self.mass_matrix, x_old, self.dt)
            step_result = _attempt(transient, self.x, self.args, self.inner)
            if step_result.success:
                break
            self.dt *= 0.25
        else:
            return self.result(False, f"pseudo-time step {step} failed down to dt = {self.dt:.2e}", step)
        self.x = step_result.x
        history.append({"dt": self.dt, "residual_norm": g_norm, "nit": step_result.nit})
        if self.verbose:
            print(f"pseudo_transient step {step}: dt={self.dt:.2e} |g|={g_norm:.2e} "
                  f"nit={step_result.nit}")
        return None


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
    run = _PseudoTransient(function=function, args=args, x=x, mass_matrix=mass_matrix, inner=inner,
                           steady=steady, dt=dt, dt_max=dt_max, max_steps=max_steps, growth=growth,
                           steady_ratio=steady_ratio, steady_every=steady_every, verbose=verbose)
    return run.march()


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
    for k in range(len(path_p) - 3, -1, -1):
        base = path_p[-1] - path_p[k]
        if base != 0 and abs(step) <= _MAX_SECANT_RATIO * abs(base):
            yield path_x[-1] + (step / base) * (path_x[-1] - path_x[k]), True
            return
    yield path_x[-1].copy(), False


class _Continuation:
    """One :func:`continuation` run: the path of solutions and those at the requested values."""

    def __init__(self, function, options, max_change, max_halvings, verbose):
        self.function, self.options, self.verbose = function, options, verbose
        self.max_change, self.max_halvings = max_change, max_halvings
        self.done, self.xs, self.results = [], [], []
        self.path_p, self.path_x = [], []

    def start(self, p, first):
        """Record the solution ``first`` at the first value ``p``."""
        self.done, self.xs, self.results = [p], [first.x.copy()], [first]
        self.path_p, self.path_x = [p], [first.x.copy()]

    def repeat(self, target):
        """A repeated value: nothing to solve."""
        self.done.append(target)
        self.xs.append(self.path_x[-1].copy())
        self.results.append(self.results[-1])

    def _jumped(self, res, guess):
        """Whether the converged ``res`` moved more than ``max_change`` from its prediction."""
        path_x = self.path_x
        if res.success and self.max_change is not None and len(self.path_p) >= 2:
            # no prediction to compare with before the second point
            size = max(float(np.max(np.abs(path_x[-1]))),
                       float(np.max(np.abs(res.x))), np.finfo(float).tiny)
            return float(np.max(np.abs(res.x - guess))) > self.max_change * size
        return False

    def correct(self, p):
        """Newton at ``p`` from each prediction in turn: ``(res, jumped, bounded)``."""
        res, jumped, bounded = None, False, False
        if p != self.path_p[-1]:            # else the sub-step is below round-off
            function, options = self.function, self.options
            for guess, bounded in _continuation_predictions(self.path_p, self.path_x, p):
                res = _attempt(lambda xv, _p=p: function(xv, _p), guess, (), options)
                jumped = self._jumped(res, guess)
                if res.success and not jumped:
                    break
        return res, jumped, bounded

    def _accept(self, p, res, target):
        """Add the solution at ``p`` to the path; whether it is the one at ``target``."""
        self.path_p.append(p)
        self.path_x.append(res.x.copy())
        if self.verbose:
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
        while level <= self.max_halvings and abs(step) * 0.5**level >= tried:
            level += 1
        return level

    def reach(self, target):
        """Sub-step along the path to ``target``; the failure result, None on success."""
        # Sub-steps are the requested step divided by 2**level. A failure
        # halves the sub-step; an accepted one doubles the next (at most to
        # the requested step), so a stretch needing short sub-steps does not
        # use up the halving budget while every sub-step succeeds.
        path_p, max_halvings = self.path_p, self.max_halvings
        step = target - path_p[-1]
        level = 0
        p = target
        while True:
            res, jumped, bounded = self.correct(p)
            if res is not None and res.success and not jumped:
                if self._accept(p, res, target):
                    return None
                level = max(level - 1, 0)
                h = step * 0.5**level
                p = target if abs(target - p) <= abs(h) else p + h
                continue
            # Halve until the sub-step is shorter than the one that failed: a
            # failure at the capped last sub-step would otherwise be retried
            # with the same prediction.
            level = self._halved(level, step, abs(p - path_p[-1]))
            if level > max_halvings or res is None:
                why = _continuation_failure(res, jumped, bounded)
                return OptimizeResult(
                    x=self.path_x[-1], success=False, params=self.done, xs=self.xs,
                    results=self.results, last_param=path_p[-1],
                    message=f"stopped at p = {path_p[-1]:g} on the way to {target:g}: {why}")
            p = path_p[-1] + step * 0.5**level


def _continuation_failure(res, jumped, bounded):
    """Why continuation stopped after the sub-step that gave ``res``."""
    if res is None:
        why = "the sub-step fell below round-off"
    elif jumped:
        why = "the corrector jumps to another solution branch"
        if bounded:
            why += "; probably a turning point (fold) of the solution branch"
    elif res.status == 4 and res.nit == 0:
        why = (f"the residual cannot be evaluated at the prediction ({res.message}); "
               "it lies outside the residual's domain")
    else:
        why = f"the corrector fails ({res.message})"
        if bounded:
            why += ("; a turning point (fold) of the solution branch, or a "
                    "corrector that cannot converge here (an inexact Jacobian, "
                    "such as NumJac's eps_jac, or a tolerance below round-off)")
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
    run = _Continuation(function, options, max_change, max_halvings, verbose)
    run.start(params[0], first)
    for target in params[1:]:
        if target == run.path_p[-1]:            # a repeated value: nothing to solve
            run.repeat(target)
            continue
        failed = run.reach(target)
        if failed is not None:
            return failed
    return OptimizeResult(x=run.path_x[-1], success=True, params=run.done, xs=run.xs, results=run.results,
                          last_param=run.path_p[-1], message="Reached every parameter value")


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
            below_lower = values < lower_bounds
            if np.any(below_lower):
                broadcasted_lower_bounds = np.broadcast_to(lower_bounds, values.shape)
                values[below_lower] = (1.0 + factor) * broadcasted_lower_bounds[
                    below_lower
                ] - factor * values[below_lower]
        if upper_bounds is not None:
            above_upper = values > upper_bounds
            if np.any(above_upper):
                broadcasted_upper_bounds = np.broadcast_to(upper_bounds, values.shape)
                values[above_upper] = (1.0 + factor) * broadcasted_upper_bounds[
                    above_upper
                ] - factor * values[above_upper]
