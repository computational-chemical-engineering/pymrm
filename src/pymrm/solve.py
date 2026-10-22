"""Nonlinear-solver utilities used by :mod:`pymrm`."""

import warnings

import numpy as np
from scipy import sparse
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve
from scipy.optimize import Bounds, OptimizeResult
from scipy.sparse import linalg

# What ``damping="auto"`` resolves to when neither a callback nor a callable
# linear solver is passed.
_AUTO_DAMPING = "guarded"

# Fixed constants of the globalisation (see the notes of ``newton``).
_LAM_FLOOR = 0.5      # smallest damping a bound alone may impose
_LAM_MIN = 1.0e-4     # the damping search ends below this damping
_LAM_NEAR = 0.2       # ... or below this one when the step is within _NEAR tolerances
_ESCAPE = True        # then take the (bounded) full step if its residual is finite
_NEAR = 100.0         # steps within _NEAR times the tolerance are taken in full
_STAGNATION_WINDOW = 5
_STAGNATION_RATIO = 0.8
_FLOOR_STEP = 1.0e-10  # outside _NEAR, a floor step is at most this relative to x
_ARMIJO_C = 1.0e-4
_BACKWARD_FLOOR = 1.0  # weight of the field scale in the backward error
_GLOBAL_FLOOR = 1.0e-6  # smallest field scale, relative to the largest unknown

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


class _SingularJacobian(Exception):
    pass


class _LinearSolverFailed(Exception):
    pass


class _LastEvaluationFailed(Exception):
    pass


def _row_scale(jac):
    """Reciprocal of the largest absolute entry of each row (1 for empty rows)."""
    if sparse.issparse(jac):
        s = abs(sparse.csr_array(jac)).max(axis=1)
        s = s.toarray() if sparse.issparse(s) else s
    else:
        s = np.max(np.abs(jac), axis=1)
    s = np.asarray(s, dtype=float).ravel()
    s[s == 0.0] = 1.0
    return 1.0 / s


class _LinearSolver:
    """Builds, per Newton iteration, a ``solve(rhs)`` for the Jacobian."""

    def __init__(self, solver, kwargs, row_scaling):
        self.kwargs = dict(kwargs or {})
        self.row_scaling = row_scaling
        self.kind = None
        if solver is None or (isinstance(solver, str) and solver in ("spsolve", "lu")):
            self.kind = "direct"
        elif isinstance(solver, str) and solver in ("cg", "bicgstab"):
            self.kind = solver
        elif isinstance(solver, str) and solver == "splu":
            self.kind = "splu"
        elif hasattr(solver, "factorize"):
            self.kind = "factorize"
        elif callable(solver):
            self.kind = "callable"
        else:
            raise ValueError("Unsupported solver method.")
        self.solver = solver

    def factorize(self, jac):
        """Return ``(solve, row_scale)`` for this Jacobian."""
        if self.kind == "direct":
            return self._direct(jac)
        if self.kind in ("cg", "bicgstab"):
            return self._iterative(jac), None
        if self.kind == "splu":
            return jac.solve, None
        if self.kind == "factorize":
            factor = self.solver.factorize(jac)
            return factor.solve, None
        solver, kwargs = self.solver, self.kwargs
        return (lambda rhs: solver(jac, rhs, **kwargs)), None

    def _direct(self, jac):
        if sparse.issparse(jac):
            a = sparse.csc_array(jac)
            r = _row_scale(a) if self.row_scaling else None
            if r is not None:
                a = sparse.csc_array(sparse.diags_array(r) @ a)
            options = {"permc_spec": "COLAMD", **self.kwargs}
            try:
                lu = linalg.splu(a, **options)
            except RuntimeError as exc:
                raise _SingularJacobian(str(exc)) from exc
            if r is None:
                return lu.solve, None
            return (lambda rhs: lu.solve(r * rhs)), r
        a = np.asarray(jac, dtype=float)
        r = _row_scale(a) if self.row_scaling else None
        if r is not None:
            a = r[:, None] * a
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LinAlgWarning)
            factor = lu_factor(a, check_finite=False)
        if np.any(np.diag(factor[0]) == 0.0):
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

    def max_step(self, x, dx):
        """Largest damping that removes at most ``tau`` of any distance to a bound."""
        lam = 1.0
        if self.lower is not None:
            m = self.lo_mask & (dx > 0.0) & (x > self.lower)
            if np.any(m):
                lam = min(lam, float(np.min(self.tau * (x[m] - self.lower[m]) / dx[m])))
        if self.upper is not None:
            m = self.hi_mask & (dx < 0.0) & (x < self.upper)
            if np.any(m):
                lam = min(lam, float(np.min(self.tau * (self.upper[m] - x[m]) / -dx[m])))
        return lam

    def project(self, trial, x):
        """Keep each component within ``1 - tau`` of its distance to a bound."""
        clipped = 0
        if self.lower is not None:
            floor = np.where(self.lo_mask,
                             self.lower + (1.0 - self.tau) * (x - np.where(self.lo_mask, self.lower, 0.0)),
                             -np.inf)
            m = trial < floor
            clipped += int(m.sum())
            trial[m] = floor[m]
        if self.upper is not None:
            ceil = np.where(self.hi_mask,
                            self.upper - (1.0 - self.tau) * (np.where(self.hi_mask, self.upper, 0.0) - x),
                            np.inf)
            m = trial > ceil
            clipped += int(m.sum())
            trial[m] = ceil[m]
        return clipped


def _field_reference(x, shape, scale):
    """Magnitude of each unknown's field: ``scale``, else the field's largest |x|.

    Fields lie along the last axis (pymrm's convention); a one-dimensional
    state is one field. No field reference is below ``_GLOBAL_FLOOR`` times
    the largest ``|x|``.
    """
    if scale is not None:
        return np.abs(scale)
    ax = np.abs(x).reshape(shape)
    overall = float(np.max(ax)) if x.size else 0.0
    if len(shape) >= 2:
        ref = np.max(ax.reshape(-1, shape[-1]), axis=0)
        # A floor at _GLOBAL_FLOOR of the largest unknown keeps a single-field
        # state stored without a field axis, (nz, nr), from being judged column
        # by column; a trace field is still judged far below its neighbours.
        ref = np.maximum(ref, _GLOBAL_FLOOR * overall)
        return np.broadcast_to(ref, shape).reshape(-1)
    return np.full(x.size, overall)


def _backward_error(g, jac, x, x_ref):
    """Componentwise backward error with a floor relative to the field scale.

    ``max_i |g_i| / (|J| (|x| + eta x_ref) + |J x - g|)_i`` with
    ``eta = _BACKWARD_FLOOR``: the measure of
    :func:`pymrm.checks.residual_check`, except that components far below the
    scale of their field (a reaction dead zone, an absent species) are judged
    against that scale instead of against their own size.
    """
    if sparse.issparse(jac):
        jac = sparse.csr_array(jac)
        abs_jac = abs(jac)
    else:
        jac = np.asarray(jac, dtype=float)
        abs_jac = np.abs(jac)
    denominator = abs_jac @ (np.abs(x) + _BACKWARD_FLOOR * x_ref) + np.abs(jac @ x - g)
    denominator = np.where(denominator > 0.0, denominator, np.finfo(float).tiny)
    return float(np.max(np.abs(g) / denominator))


def _is_matrix(jac):
    """Whether ``jac`` is a numeric matrix (not a factorisation or user object)."""
    if sparse.issparse(jac):
        return True
    return isinstance(jac, np.ndarray) and jac.dtype.kind in "fiu"


def _rms(v):
    return float(np.sqrt(np.mean(np.square(v))))


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
    stagnation=True,
    row_scaling=False,
    recoverable=(FloatingPointError, OverflowError, ZeroDivisionError),
    verbose=0,
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
        a lower bound. No step removes more than ``tau`` of the distance of a
        component to its bound (with a floor of 0.5 on ``lam``, below which
        the components are projected individually). A component may sit on
        its bound; an initial guess outside is projected, with a warning.
        Use bounds where the residual is undefined (``T > 0``, arguments of
        ``log`` or ``sqrt``), not to enforce physical positivity: see Notes.
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
        Stop at the stagnation floor (default True): when the step test cannot
        be met because round-off, or switching terms such as ``max(c, 0)`` and
        upwinding, keep the steps from shrinking, a full step no smaller than
        0.8 times the smallest of the five before it ends the iteration with
        success, provided ``backward_tol`` passes; otherwise the iteration
        continues. It applies to steps within 100 tolerances, and to steps below
        1e-10 of ``|x|`` once ``backward_tol`` passes.
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
        2 ``maxfev`` reached; 3 damping fell below its minimum; 4 non-finite
        residual; 5 singular Jacobian; 6 linear solver failed; 7 step test
        met but ``backward_error > backward_tol``; 8 stopped on a bound (the
        bound is likely cutting off the solution).

    Raises
    ------
    ValueError
        If ``solver`` or ``damping`` is not supported.

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

    **Changes after 2.5.0.** Before, ``newton`` took plain full steps even
    into a non-finite residual, stopped on the step test alone, and switched
    to ILU-BiCGSTAB above 50 000 unknowns. Pass ``damping="none",
    backward_tol=None, stagnation=False`` for the old iteration; it then
    takes the same steps and evaluations as before. A singular Jacobian or a
    non-finite residual now ends the solve with a status instead of
    continuing with NaN.
    """
    x = np.array(initial_guess, dtype=float, order="C")
    shape = x.shape
    n = x.size
    x_flat = x.reshape(-1)  # view: updates of x_flat update x

    if lin_solver_kwargs is None:
        lin_solver_kwargs = {}
    lin = _LinearSolver(solver, lin_solver_kwargs, row_scaling)

    if damping == "auto":
        user_hooks = callback is not None or lin.kind == "callable"
        mode = "none" if user_hooks else _AUTO_DAMPING
    elif damping in ("none", "guarded", "monotone", "armijo"):
        mode = damping
    else:
        raise ValueError(f"Unsupported damping {damping!r}.")

    d_user = None if scale is None else \
        np.array(np.broadcast_to(np.asarray(scale, dtype=float), shape)).ravel()
    if d_user is not None and not (np.all(np.isfinite(d_user)) and np.all(d_user > 0.0)):
        raise ValueError("scale must be finite and positive for every unknown.")
    d = np.ones(n) if d_user is None else d_user
    relative = rtol > 0 or not np.isscalar(tol)
    tol_vec = np.array(np.broadcast_to(np.asarray(tol, dtype=float), shape)).ravel() \
        if relative else None
    bnd = _Bounds(*_parse_bounds(bounds, shape), tau)
    if bnd.active:
        bnd.enforce_initial(x_flat)
    use_stagnation = stagnation and refresh is None

    nfev = 0
    nfact = 0
    history = []
    state = {"last": False, "ev": None, "step": float("nan")}

    def evaluate(xv, guard):
        """Residual (flat) and Jacobian at ``xv``; None if unusable and guarded."""
        nonlocal nfev
        nfev += 1
        if guard:
            try:
                g_raw, jac = function(xv, *args)
            except recoverable:
                return None
            except Exception as exc:
                if state["last"]:
                    # The old iteration never evaluated here: report, do not raise.
                    raise _LastEvaluationFailed(f"{type(exc).__name__}: {exc}") from exc
                raise
        else:
            g_raw, jac = function(xv, *args)
        g_raw = g_raw.toarray() if sparse.issparse(g_raw) else np.array(g_raw, dtype=float)
        g_flat = g_raw.reshape(-1)
        finite = np.all(np.isfinite(g_flat))
        if finite and _is_matrix(jac):
            data = jac.data if sparse.issparse(jac) else jac
            finite = np.all(np.isfinite(data))
        if not finite:
            return None
        return g_raw, g_flat, jac

    def backward_error(g_flat, jac):
        if not _is_matrix(jac):
            return float("nan")
        return _backward_error(g_flat, jac, x_flat, _field_reference(x_flat, shape, d_user))

    def finish(status, success, ev, step_norm, be, extra=""):
        g_raw = None if ev is None else ev[0]
        message = _MESSAGES[status] + (f": {extra}" if extra else "")
        return OptimizeResult(x=x, success=bool(success), status=status, message=message,
                              nit=len(history), nfev=nfev, nfact=nfact, fun=g_raw,
                              step_norm=step_norm, backward_error=be, history=history)

    ev = evaluate(x, guard=False)
    if ev is None:
        return finish(4, False, None, float("nan"), float("nan"), "at the initial guess")

    def iterate():
        nonlocal nfact, ev
        lam_next = 1.0
        window = []          # step norms of the recent full steps at the floor
        step_inf = float("nan")
        be = float("nan")
        gate_failed = False
        clipped = 0

        def final_step(status, extra=""):
            """Take the converged step without evaluating there, as plain Newton did."""
            trial = x_flat - dx
            n_clip = bnd.project(trial, x_flat) if bnd.active else 0
            x_flat[:] = trial
            history.append({"step_norm": step_inf, "damping": 1.0, "theta": float("nan"),
                            "clipped": n_clip})
            if callback is not None:
                callback(x, g_raw)
            if refresh is not None:
                refresh(x)
            return finish(status, True, ev, step_inf, be, extra)

        for it in range(int(maxfev)):
            g_raw, g, jac = ev
            state["ev"] = ev
            last_it = it == int(maxfev) - 1
            try:
                solve, r = lin.factorize(jac)
                nfact += 1
                dx = np.asarray(solve(g_raw if lin.kind == "callable" else g),
                                dtype=float).reshape(-1)
            except _SingularJacobian as exc:
                return finish(5, False, ev, step_inf, float("nan"), str(exc))
            except _LinearSolverFailed as exc:
                return finish(6, False, ev, step_inf, float("nan"), str(exc))
            if not np.all(np.isfinite(dx)):
                return finish(6, False, ev, step_inf, float("nan"), "non-finite Newton step")

            dy = dx / d
            step_inf = float(np.max(np.abs(dy))) if n else 0.0
            y = x_flat / d
            if relative:
                limit = tol_vec + rtol * np.abs(y)
                converged = bool(np.all(np.abs(dy) <= limit))
                with np.errstate(divide="ignore", invalid="ignore"):
                    excess = float(np.max(np.where(dy == 0.0, 0.0, np.abs(dy) / limit)))
            else:
                converged = step_inf < tol
                excess = step_inf / tol if tol > 0 else (0.0 if step_inf == 0 else np.inf)
            converged = converged or step_inf == 0.0
            near = converged or (refresh is None and excess < _NEAR)
            state["step"] = step_inf
            state["last"] = converged or last_it

            # The verdict uses the residual at x, before the step: no evaluation
            # beyond those of plain Newton. Near a solution the step only improves it.
            be = float("nan")
            if near or last_it or relative:
                be = backward_error(g, jac)
            at_floor = near
            if use_stagnation and not near and backward_tol is not None:
                # tol may lie below the round-off floor by more than _NEAR: once the
                # backward error passes, non-shrinking steps are that floor too. The
                # step must also be round-off small relative to x: a nearly singular
                # system has a tiny backward error at a huge x.
                y_size = float(np.max(np.abs(y))) if n else 0.0
                at_floor = step_inf <= _FLOOR_STEP * y_size
                if at_floor:
                    be = backward_error(g, jac)
                    at_floor = be <= backward_tol
            gate = backward_tol is None or np.isnan(be) or be <= backward_tol
            stagnant = (use_stagnation and at_floor and len(window) >= _STAGNATION_WINDOW
                        and step_inf >= _STAGNATION_RATIO * min(window[-_STAGNATION_WINDOW:]))

            theta = float("nan")
            if mode == "none":
                # Plain (or bound-shortened) step; callback and refresh before the
                # next evaluation, as the callers of the undamped iteration expect.
                lam = 1.0
                if not near and bnd.active:
                    lam = max(_LAM_FLOOR, bnd.max_step(x_flat, dx))
                trial = x_flat - lam * dx
                clipped = bnd.project(trial, x_flat) if bnd.active else 0
                x_flat[:] = trial
                history.append({"step_norm": step_inf, "damping": lam, "theta": theta,
                                "clipped": clipped})
                if callback is not None:
                    callback(x, g_raw)
                if refresh is not None:
                    refresh(x)
                if relative and step_inf > 0.0:
                    # The relative test uses x after the step (and the callback).
                    converged = bool(np.all(np.abs(dy) <= tol_vec + rtol * np.abs(x_flat / d)))
                if converged and gate:
                    return finish(0, True, ev, step_inf, be)
                if stagnant and gate:
                    if clipped:
                        return finish(8, False, ev, step_inf, be,
                                      f"{clipped} components held by a bound")
                    return finish(1, True, ev, step_inf, be,
                                  f"|dx| = {step_inf:.2e} (tol {np.max(tol):.1e})")
                gate_failed = converged
                if last_it:
                    break
                ev_new = evaluate(x, guard=False)
                if ev_new is None:
                    return finish(4, False, ev, step_inf, float("nan"), "after a full step")
                ev = ev_new
            else:
                if converged and gate:
                    return final_step(0)
                if stagnant and gate:
                    if clipped:
                        return finish(8, False, ev, step_inf, be,
                                      f"{clipped} components held by a bound")
                    return final_step(1, f"|dx| = {step_inf:.2e} (tol {np.max(tol):.1e})")
                gate_failed = converged
                accepted = None
                lam_max = max(_LAM_FLOOR, bnd.max_step(x_flat, dx)) if bnd.active else 1.0
                at_lam_max = None   # (trial, ev_t, clipped) of a trial at lam_max, if made
                if mode == "guarded":
                    # Full step (shortened only by the bounds far from the solution).
                    lam = 1.0 if near else lam_max
                    trial = x_flat - lam * dx
                    clipped = bnd.project(trial, x_flat) if bnd.active else 0
                    ev_t = evaluate(trial.reshape(shape), guard=True)
                    if lam == lam_max:
                        at_lam_max = (trial, ev_t, clipped)
                    if ev_t is not None:
                        accepted = (trial, ev_t, lam)
                    else:
                        lam_next = 0.25 * lam
                if accepted is None:
                    # Damped search: monotonicity test or Armijo, from lam_next.
                    lam = min(lam_next, lam_max)
                    dy_rms = _rms(dy)
                    row = r if r is not None else (_row_scale(jac) if _is_matrix(jac) else 1.0)
                    phi0 = float(np.linalg.norm(row * g))
                    best = None
                    # Near the solution a failed test is usually round-off: search briefly.
                    lam_min = _LAM_NEAR if near else _LAM_MIN
                    while lam >= lam_min:
                        trial = x_flat - lam * dx
                        clipped = bnd.project(trial, x_flat) if bnd.active else 0
                        ev_t = evaluate(trial.reshape(shape), guard=True)
                        if lam == lam_max:
                            at_lam_max = (trial, ev_t, clipped)
                        if ev_t is None:
                            lam *= 0.25
                            continue
                        if mode == "armijo":
                            phi = float(np.linalg.norm(row * ev_t[1]))
                            if phi <= (1.0 - _ARMIJO_C * lam) * phi0:
                                accepted = (trial, ev_t, lam)
                                break
                            if best is None or phi < best[0]:
                                best = (phi, trial, ev_t, lam, clipped)
                            lam *= 0.5
                            continue
                        try:
                            rhs = ev_t[0] if lin.kind == "callable" else ev_t[1]
                            dbar = np.asarray(solve(rhs), dtype=float).reshape(-1) / d
                        except _LinearSolverFailed as exc:
                            return finish(6, False, ev, step_inf, float("nan"), str(exc))
                        theta = _rms(dbar) / dy_rms if dy_rms > 0.0 else 0.0
                        if theta <= 1.0 - lam / 4.0:
                            accepted = (trial, ev_t, lam)
                            break
                        mu = 0.5 * dy_rms * lam**2 / max(_rms(dbar - (1.0 - lam) * dy), 1.0e-300)
                        lam = min(0.5 * lam, max(mu, 0.1 * lam))
                    if accepted is None and best is not None and best[0] < phi0:
                        _, trial, ev_t, lam, clipped = best
                        ev_t = evaluate(trial.reshape(shape), guard=True)  # restore model state
                        if ev_t is not None:
                            accepted = (trial, ev_t, lam)
                    escaped = False
                    if accepted is None and _ESCAPE:
                        lam = lam_max
                        if at_lam_max is not None:
                            trial, ev_t, clipped = at_lam_max
                            if ev_t is not None:
                                ev_t = evaluate(trial.reshape(shape), guard=True)
                        else:
                            trial = x_flat - lam * dx
                            clipped = bnd.project(trial, x_flat) if bnd.active else 0
                            ev_t = evaluate(trial.reshape(shape), guard=True)
                        if ev_t is not None:
                            accepted = (trial, ev_t, lam)
                            escaped = True
                    if accepted is None:
                        extra = f"{lam_min:g} at iteration {it}"
                        if clipped:
                            extra += f"; {clipped} components held by a bound"
                        return finish(3, False, ev, step_inf, backward_error(g, jac), extra)
                    lam_next = 1.0 if (escaped or mode == "guarded") else min(1.0, 4.0 * accepted[2])

                trial, ev, lam = accepted
                x_flat[:] = trial
                history.append({"step_norm": step_inf, "damping": lam, "theta": theta,
                                "clipped": clipped})
                stale = False
                if callback is not None:
                    # The callback may change x or model state: re-evaluate after it.
                    callback(x, g_raw)
                    stale = True
                if refresh is not None:
                    refresh(x)
                    stale = True
                if stale:
                    ev_new = evaluate(x, guard=True)
                    if ev_new is None:
                        return finish(4, False, ev, step_inf, float("nan"),
                                      "after the callback or refresh")
                    ev = ev_new
            if verbose:
                print(f"newton it={it:3d} |dx|={step_inf:.3e} lam={lam:.3g} "
                      f"theta={theta:.3g} clipped={clipped} nfev={nfev}")
            if stagnant:
                window.clear()      # the backward error did not pass: keep iterating
            elif use_stagnation and at_floor and lam == 1.0:
                window.append(step_inf)
            else:
                window.clear()

        if gate_failed:
            return finish(7, False, ev, step_inf, be, f"{be:.2e} after maxfev iterations")
        if clipped:
            return finish(8, False, ev, step_inf, be, f"{clipped} components held by a bound")
        return finish(2, False, ev, step_inf, be)

    try:
        return iterate()
    except _LastEvaluationFailed as exc:
        return finish(2, False, state["ev"], state["step"], float("nan"),
                      f"the residual at the last trial point raised {exc}")


def _mass_matrix(mass, shape):
    """Sparse matrix of the pseudo-time accumulation term."""
    if sparse.issparse(mass):
        return sparse.csc_array(mass)
    diag = np.array(np.broadcast_to(np.asarray(mass, dtype=float), shape)).ravel()
    return sparse.diags_array(diag).tocsc()


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
    history = []
    steady_result = None
    g_prev = None
    g_attempt = None

    def result(success, message, nsteps):
        xr = steady_result.x if (success and steady_result is not None) else x
        return OptimizeResult(x=xr, success=success, message=message, nsteps=nsteps,
                              history=history, steady=steady_result)

    for step in range(int(max_steps) + 1):
        g_now, _ = function(x, *args)
        g_norm = float(np.linalg.norm(np.asarray(g_now, dtype=float).ravel()))
        if not np.isfinite(g_norm):
            return result(False, "non-finite residual during the pseudo-time march", step)
        if (g_attempt is None or g_norm <= steady_ratio * g_attempt or step == max_steps
                or (steady_every and step % steady_every == 0)):
            g_attempt = g_norm
            steady_result = newton(function, x, args, **steady)
            if steady_result.success:
                return result(True, f"Converged by the steady Newton solve after {step} "
                              "pseudo-time steps", step)
            if step == max_steps:
                break
        if g_prev is not None and g_norm > 0.0 and history and history[-1]["nit"] <= 3:
            dt = min(dt_max, dt * min(growth, max(2.0, g_prev / g_norm)))
        g_prev = g_norm
        x_old = x.ravel().copy()
        for _ in range(20):
            def transient(xv, *a, _dt=dt):
                g, jac = function(xv, *a)
                g = np.asarray(g, dtype=float).ravel() + mass_matrix @ (xv.ravel() - x_old) / _dt
                return g, jac + mass_matrix / _dt
            step_result = newton(transient, x, args, **inner)
            if step_result.success:
                break
            dt *= 0.25
        else:
            return result(False, f"pseudo-time step {step} failed down to dt = {dt:.2e}", step)
        x = step_result.x
        history.append({"dt": dt, "residual_norm": g_norm, "nit": step_result.nit})
        if verbose:
            print(f"pseudo_transient step {step}: dt={dt:.2e} |g|={g_norm:.2e} "
                  f"nit={step_result.nit}")
    return result(False, "max_steps reached without a converged steady solve", int(max_steps))


def continuation(function, initial_guess, params, *, max_halvings=6, max_change=0.2,
                 newton_kwargs=None, verbose=0):
    """Solve ``function(x, p) = 0`` along a sequence of parameter values.

    Natural-parameter continuation: each point starts from a secant
    extrapolation of the two previous solutions and is corrected with
    :func:`newton`. A failed correction halves the parameter step (at most
    ``max_halvings`` times per requested value). A correction that moves
    ``x`` by more than ``max_change`` times ``max |x|`` from the prediction
    counts as failed: near a turning point (fold) the corrector otherwise
    jumps silently to another solution branch. The march then stops and
    reports where; the fold lies between the last parameter reached and the
    next one. Following a branch around a fold needs arclength continuation,
    which this function does not do.

    Parameters
    ----------
    function : callable
        ``function(x, p) -> (residual, jacobian)``.
    initial_guess : numpy.ndarray
        Guess for the first parameter value.
    params : sequence of float
        Parameter values, in the order to visit them.
    max_halvings : int, optional
        Largest number of step halvings on the way to one parameter value.
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
    first = newton(lambda xv: function(xv, params[0]), np.array(initial_guess, dtype=float),
                   **options)
    if not first.success:
        return OptimizeResult(x=first.x, success=False, params=[], xs=[], results=[first],
                              last_param=None,
                              message=f"no solution at the first value {params[0]:g}: "
                                      f"{first.message}")
    done, xs, results = [params[0]], [first.x.copy()], [first]
    path_p, path_x = [params[0]], [first.x.copy()]
    for target in params[1:]:
        halvings = 0
        p = target
        while True:
            if len(path_p) >= 2:
                ratio = (p - path_p[-1]) / (path_p[-1] - path_p[-2])
                guess = path_x[-1] + ratio * (path_x[-1] - path_x[-2])
            else:
                guess = path_x[-1].copy()
            res = newton(lambda xv, _p=p: function(xv, _p), guess, **options)
            jumped = False
            if res.success and max_change is not None:
                size = max(float(np.max(np.abs(path_x[-1]))), np.finfo(float).tiny)
                jumped = float(np.max(np.abs(res.x - guess))) > max_change * size
            if res.success and not jumped:
                path_p.append(p)
                path_x.append(res.x.copy())
                if verbose:
                    print(f"continuation p={p:g} nit={res.nit}")
                if p == target:
                    done.append(target)
                    xs.append(res.x.copy())
                    results.append(res)
                    break
                p = target
                continue
            halvings += 1
            if halvings > max_halvings:
                why = ("the corrector jumps to another solution branch" if jumped
                       else f"the corrector fails ({res.message})")
                return OptimizeResult(
                    x=path_x[-1], success=False, params=done, xs=xs, results=results,
                    last_param=path_p[-1],
                    message=f"stopped at p = {path_p[-1]:g} on the way to {target:g}: {why}; "
                            "probably a turning point (fold) of the solution branch")
            p = path_p[-1] + 0.5 * (p - path_p[-1])
    return OptimizeResult(x=path_x[-1], success=True, params=done, xs=xs, results=results,
                          last_param=path_p[-1], message="Reached every parameter value")


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
