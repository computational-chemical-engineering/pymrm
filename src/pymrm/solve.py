"""Nonlinear-solver utilities used by :mod:`pymrm`."""

import warnings

import numpy as np
from scipy import sparse
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve
from scipy.optimize import Bounds, OptimizeResult
from scipy.sparse import linalg

# What ``damping="auto"`` resolves to when neither a callback nor a callable
# linear solver is passed.
_AUTO_DAMPING = "none"

# Fixed constants of the globalisation (see the notes of ``newton``).
_LAM_FLOOR = 0.5      # smallest damping a bound alone may impose
_LAM_MIN = 1.0e-4     # the damping search ends below this damping
_LAM_NEAR = 0.2       # ... or below this one when the step is within _NEAR tolerances
_ESCAPE = True        # then take the (bounded) full step if its residual is finite
_NEAR = 100.0         # steps within _NEAR times the tolerance are taken in full
_STAGNATION_WINDOW = 5
_STAGNATION_RATIO = 0.8
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
    backward_tol=None,
    stagnation=False,
    row_scaling=True,
    recoverable=(FloatingPointError, OverflowError, ZeroDivisionError),
    verbose=0,
):
    """Solve ``function(x) = 0`` with a (damped) Newton iteration.

    Every iteration solves ``J dx = g`` with a direct sparse LU factorisation
    and updates ``x <- x - lam dx``. With the default options the step length
    ``lam`` is 1 (plain Newton). For hard problems, ``damping``, ``bounds`` and
    ``scale`` make the iteration robust far from the solution, and
    ``backward_tol`` makes the convergence verdict independent of the units of
    the unknowns; see the Notes.

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
        * ``'auto'``: currently ``'none'``. It stays ``'none'`` whenever a
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
    backward_tol : float, optional
        If given, the step test is necessary but not sufficient: the backward
        error at the new ``x`` must also be below ``backward_tol``, else the
        iteration continues. The backward error is
        ``max_i |g_i| / (|J| (|x| + x_ref) + |J x - g|)_i``, the relative
        change of the equation coefficients that would make ``x`` exact, with
        ``x_ref`` the magnitude of each unknown's field: ``scale`` if given,
        else the largest ``|x|`` of that field (fields along the last axis; a
        one-dimensional state is one field). Components far below their
        field's magnitude, such as a reaction dead zone, are thus judged
        against the field, while a field that is small everywhere (a trace
        species) is judged against itself. It is always reported in the
        result. For a flat state that mixes units, pass ``scale``.
    stagnation : bool, optional
        Stop at the stagnation floor: once steps are taken in full near the
        solution, a step no smaller than 0.8 times the smallest of the five
        before it ends the iteration (round-off, or switching terms such as
        ``max(c, 0)`` and upwinding, keep the steps from shrinking further).
        Whether that counts as success is decided by ``backward_tol``.
    row_scaling : bool, optional
        Divide each row of the Jacobian by its largest entry before the direct
        LU factorisation. Improves pivoting when rows have different units.
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
        evaluations), ``nfact`` (factorisations), ``fun`` (the residual at
        ``x``, as returned by ``function``), ``step_norm`` (infinity norm of
        the last Newton step), ``backward_error`` (at ``x``; ``nan`` when the
        Jacobian is a factorisation), and ``history``, a list with per
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

    **Convergence verdict.** The default step test is absolute. A trace
    unknown much smaller than ``tol`` passes it immediately. ``backward_tol``
    (for example ``1e-8``) guards against that without changing the meaning
    of ``tol``; :func:`pymrm.checks.residual_check` gives a stricter,
    purely componentwise verdict after the fact.
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

    def evaluate(xv, guard):
        """Residual (flat) and Jacobian at ``xv``; None if unusable and guarded."""
        nonlocal nfev
        nfev += 1
        if guard:
            try:
                g_raw, jac = function(xv, *args)
            except recoverable:
                return None
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

    lam_next = 1.0
    window = []
    step_inf = float("nan")
    gate_failed = False
    clipped = 0

    for it in range(int(maxfev)):
        g_raw, g, jac = ev
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
            last = converged or it == int(maxfev) - 1
            try:
                ev_new = evaluate(x, guard=last)
            except Exception:
                if not last:
                    raise
                ev_new = None
            if ev_new is None:
                if converged:
                    # The step test decides, as before, even when the residual
                    # cannot be evaluated at the new point.
                    return finish(0, True, ev, step_inf, float("nan"),
                                  "the residual at x could not be evaluated")
                if last:
                    return finish(2, False, ev, step_inf, float("nan"),
                                  "the residual at x could not be evaluated")
                return finish(4, False, ev, step_inf, float("nan"), "after a full step")
            ev = ev_new
        else:
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
                row = r if r is not None else 1.0
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
        stagnant = False
        at_floor = near
        if use_stagnation and not near and lam == 1.0 and backward_tol is not None:
            # tol may lie below the round-off floor by more than _NEAR: once the
            # backward error passes, non-shrinking full steps are that floor too.
            at_floor = backward_error(ev[1], ev[2]) <= backward_tol
        if use_stagnation and at_floor and lam == 1.0:
            stagnant = (len(window) >= _STAGNATION_WINDOW
                        and step_inf >= _STAGNATION_RATIO * min(window[-_STAGNATION_WINDOW:]))
            window.append(step_inf)
        else:
            window.clear()

        if converged or stagnant:
            be = backward_error(ev[1], ev[2])
            gate = backward_tol is None or be <= backward_tol
            if converged and gate:
                return finish(0, True, ev, step_inf, be)
            if stagnant:
                if clipped:
                    return finish(8, False, ev, step_inf, be,
                                  f"{clipped} components held by a bound")
                if gate:
                    return finish(1, True, ev, step_inf, be,
                                  f"|dx| = {step_inf:.2e} (tol {np.max(tol):.1e})")
                return finish(7, False, ev, step_inf, be, f"{be:.2e} at the stagnation floor")
            gate_failed = True
        else:
            gate_failed = False

    be = backward_error(ev[1], ev[2])
    if gate_failed:
        return finish(7, False, ev, step_inf, be, f"{be:.2e} after maxfev iterations")
    if clipped:
        return finish(8, False, ev, step_inf, be, f"{clipped} components held by a bound")
    return finish(2, False, ev, step_inf, be)


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
