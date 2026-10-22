# Plan: a robust and efficient `pymrm.newton`

Status: plan, agreed in outline on 2026-10-22. The open decisions of section 7 were
taken provisionally as recommended, so the work could proceed overnight; each is
marked "provisional" and can be reversed at review.

## 1. Why

`pymrm.newton` takes plain full Newton steps, stops on an absolute step size, and
switches silently from a direct solver to ILU-preconditioned BiCGSTAB above 50 000
unknowns. Two research repositories had to write their own solver on top of it:

| Added in a project | Where | Failure without it |
|---|---|---|
| Damping by the natural monotonicity test, positivity bound | `pcec-membrane-reactor/src/pcec/solver.py` (`solve_bounded`) | a full step drives a concentration negative, clipped kinetics lose the restoring term, the solve ends in NaN; operating-map coverage 11/20 without, 20/20 with |
| Armijo line search on a scaled residual, backtracking on numerical exceptions | `ammonia_synthesis_reactor/src/reactor/solvers.py` | aborts instead of taking a shorter step |
| Row and column scaling | pcec | energy and mass rows differ by orders of magnitude |
| Stop at the stagnation floor | pcec | `max(c, 0)` and upwind switching create a round-off floor above `tol` |
| `solver="spsolve"` pinned explicitly | gallery D1.1, all pcec calls | the automatic BiCGSTAB switch |
| Way to detect false convergence | pitfall P7, `test_checks.py:64-70` | an absolute `tol` returns a wrong answer with `success=True` for trace species |

Goal: one solver that is efficient on easy problems, robust on hard ones, documents
its options for hard problems, and keeps every existing call working with a
reasonable (not necessarily bit-identical) result.

## 2. Existing use that must keep working

Survey of about 250 calls in the pymrm tests and exemplars, pymrm-gallery (87 files),
pymrm-book lecture notes, pymrm-book-teacher solutions and pcec:

- Nearly all calls pass only `maxfev` and `tol`; about 8 use `tol=0` with `rtol`.
- `callback=clip_approach` at about 25 gallery sites and several teacher solutions,
  some with upper bounds. These rely on the in-place projection after the step.
- Other callbacks: a lagged velocity update (pcec `model.py:1111`), a stateful
  `limit_step` (book `L8_pressure_velocity.ipynb:885`).
- `solver=callable`: a direction-capping solver (gallery I1.2
  `build_page.py:504`), a Schur-complement solve (teacher).
- `'cg'`, `'bicgstab'`, `'splu'` (pre-factorised Jacobian), `lin_solver_kwargs`,
  `args` and an array `tol` appear only in pymrm tests.
- Dense Jacobians are used.
- Callers read almost only `.x` and `.success`. `.fun` is currently the residual
  before the last update.
- No known caller exceeds 50 000 unknowns.

Pinned by tests and docs: `test/test_solve.py:16-190`, `test/test_pitfalls.py:177-192`
(P7), `test/test_checks.py:64-70`; `plugins/pymrm/skills/conventions/references/`
`pitfalls.md:104-120`, `api-map.md:30-31`, `style-guide.md:443-463` and `:594`,
`model-patterns/SKILL.md:87`. The rule in `AGENTS.md` applies: a behaviour change
described by a pitfall updates `pitfalls.md` and its test in the same change.

## 3. Design

### Signature

```python
def newton(function, initial_guess, args=(), tol=1.49012e-08, maxfev=100,
           solver=None, lin_solver_kwargs=None, callback=None, rtol=0.0, *,
           damping="auto", bounds=None, tau=0.99, scale=None, refresh=None,
           backward_tol=None, stagnation=False, row_scaling=True,
           recoverable=(FloatingPointError, OverflowError, ZeroDivisionError),
           verbose=0)
```

All new options are keyword-only, so positional calls are unaffected. Keyword
arguments are preferred over an options object: the docstring shows every default.

### Linear solver

- Default: direct LU at every size. Sparse Jacobians use `scipy.sparse.linalg.splu`
  with COLAMD, dense ones `scipy.linalg.lu_factor`. The factorisation is reused for
  the simplified Newton correction of the damping test. No size switch.
- `row_scaling=True`: rows equilibrated by their largest entry before factorising
  (`splu` does not equilibrate). Changes iterates only at round-off level and
  improves pivoting across mixed-unit rows.
- Iterative solvers only on explicit request (`'cg'`, `'bicgstab'`, or a solver
  object).
- Kept: `'spsolve'` (alias of the default, so existing pins still mean direct),
  `'splu'` with a pre-factorised Jacobian, the callable `(J, rhs, **kw)` signature.
- New: a factorize protocol, an object whose `factorize(J)` returns something with
  `.solve(rhs)` (a `SuperLU` qualifies). Custom solvers such as a Schur complement
  can then take part in factorisation reuse and in the monotonicity test.
- Not now: reusing the Jacobian across iterations (chord). While `function` returns
  `(g, J)` together it saves the factorisation but not the assembly; revisit when a
  measured case shows factorisation dominates.

### Globalisation (`damping`)

- `"none"`: full steps, current behaviour.
- `"monotone"`: Deuflhard's natural monotonicity test, as in pcec `solve_bounded`:
  accept damping `lam` when the simplified correction satisfies
  `theta = ||dbar|| / ||dx|| <= 1 - lam/4`; otherwise reduce using the predicted
  `mu`, at least by a factor 2 and at most by 10; next iteration starts at
  `min(1, 4 lam)`. Affine covariant, hence independent of residual row scaling.
  Costs one back-substitution per trial; a full step accepted at `lam = 1` reuses
  its evaluation as the next iterate.
- `"armijo"`: backtracking on the row-equilibrated residual norm, keeping the best
  finite trial, for cases where the simplified correction is meaningless.
- `"auto"`: `"none"` in stage 1, `"monotone"` from stage 2, but stays `"none"`
  when a `callback` or a callable `solver` is passed (these sites modify the step
  after the fact or are stateful; see decision 1).
- Trial points whose residual is not finite, or whose evaluation raises one of
  `recoverable`, are rejected and the damping divided by 4. Exceptions at the
  initial guess propagate. `ValueError` is not in the default tuple because shape
  bugs raise it.
- Give up when the damping falls below `lam_min = 1e-6`, or when a residual is
  non-finite in `"none"` mode (today that iterates NaN until `maxfev`).

### Bounds

- `bounds=(lower, upper)`. Each side is `None`, a scalar, or an array broadcast
  against the shape of `initial_guess` (not the flattened vector): shape `(nc,)`
  gives one bound per field, `(nz, 1)` one per cell, the full shape one per
  component. `-np.inf` / `np.inf` leaves a component unbounded. Shorthand
  `bounds=0.0` means `(0.0, None)`; a `scipy.optimize.Bounds` object is accepted.
- Each trial point is bounded before it is evaluated, so the damping test judges
  the point that is kept: fraction to the boundary (no step removes more than
  `tau` of the distance to a finite bound), with a floor of 0.5 on the step length
  so one cell cannot stall the solve, and componentwise projection below that floor.
- A component may sit exactly on its bound. An `initial_guess` outside the bounds
  is projected once, with a warning.
- When the bound cuts off the solution (the step keeps pointing outward), the
  solve ends with its own status, not with success.
- Bounds are meant for where the residual is undefined, not for physical
  positivity; see section 4.

### Scaling and lagged quantities

- `scale`: column scaling; the iteration runs in `y = x / scale` and `tol`, `rtol`
  apply to the scaled step.
- `refresh(x)`: called after every accepted iterate, for quantities updated outside
  the Jacobian (lagged velocities). The function is then re-evaluated. With
  `refresh` the monotonicity test is approximate and the stagnation rule is off.

### Stopping

- The step test keeps its exact current meaning: `||dx||_inf < tol`, or
  `|dx| <= tol + rtol |x|` componentwise when `rtol > 0` or `tol` is an array.
- New gate `backward_tol`: the scale-free backward error of
  `pymrm.checks.residual_check`, evaluated at the final `x`. The step test becomes
  necessary but not sufficient; if the gate fails the iteration continues, reusing
  that evaluation. This removes the P7 trap without changing what a caller's
  physical `tol` means.
- `stagnation`: once steps are within `100 tol` full steps are taken; when the step
  is no smaller than 0.8 times the smallest of the previous five, the iteration
  stops at the floor. Whether that is success is decided by the gate. Off with
  `refresh`.
- `callback(x, g)` keeps its contract: called once per accepted iterate, may modify
  `x` in place. A modification is detected and the function re-evaluated.

### Result

`scipy.optimize.OptimizeResult` with:

- `x`, `success`, `nit`, `message`, `step_norm` (as now; `maxfev` still counts
  Newton iterations, documented as such);
- `fun`: the residual at `x` (changed; see decision 3);
- `status`: 0 converged, 1 stagnation floor with gate passed, 2 `maxfev` reached,
  3 damping below `lam_min`, 4 non-finite residual, 5 singular Jacobian, 6 linear
  solver failed, 7 step test met but gate failed, 8 stopped on a bound;
- `nfev`, `nfact`, `backward_error`;
- `history`: per iteration the step norm, damping, `theta` and backward error.

### Kept out of the library

- Problem-specific escalation (the ammonia temperature-continuation ladder).
- Arclength continuation, Newton-Krylov, Broyden.
- The ammonia divergence guard `||g|| > 10 ||g0||`: the monotonicity test with
  `lam_min` replaces it.

### Helpers (stage 3)

- `pseudo_transient(function, x0, mass, dt0, ...)`: implicit Euler in pseudo-time
  with switched evolution relaxation growth of `dt`, each step solved by `newton`.
  The caller supplies `mass`; the library cannot know the units of the time term.
- `continuation(fun_of_p, x0, params, ...)`: natural-parameter continuation with a
  secant predictor and step halving, one result per point. It does not trace folds.

## 4. Kinetics near zero concentration: regularise instead of bounding

A hard lower bound at zero often hurts convergence:

- The discrete solution can itself be slightly negative near steep fronts (central
  differences, TVD and higher-order convection). A bound at zero excludes it and the
  iteration stalls against the bound.
- `max(c, 0)` in a rate makes its derivative zero below zero. The Jacobian loses the
  term that pulls `c` back and the next step overshoots further. This was the pcec
  failure mode; the bounded solver treated the symptom.
- `c^n` with `n < 1` has an infinite slope at zero.

Recommended instead: write kinetics that are smooth and defined for slightly
negative concentrations, and first order near zero, so a negative concentration
gives a restoring rate.

| Kinetics | Regularised form | Behaviour |
|---|---|---|
| power law `c^n`, `n < 1` | `k c (c**2 + eps**2)**((n - 1)/2)` | about `k eps^(n-1) c` near zero, odd through zero; `k c^n` for `c >> eps` |
| Langmuir-Hinshelwood `c/(1 + K c)` | `c / (1 + K abs(c))` | denominator stays positive |
| reversible `k (c_A - c_B/K)` | unchanged | already linear |

`eps` is a small fraction of a reference concentration (for example `1e-6 c_ref`).
It changes the model where `c` is of order `eps`, which for `n < 1` is the
dead-zone edge, so check that results do not change when `eps` is reduced tenfold.

Consequences for the library and its guidance:

1. Bounds are documented as the domain where the residual is defined: `T > 0`
   (Arrhenius, ideal-gas density), arguments of `log` and `sqrt` (Nernst
   potentials, chemical potentials). Not for physical positivity as such.
2. A guard bound below zero, for example `bounds=(-1e-3 * c_ref, None)`, prevents
   wild overshoot while allowing small undershoots. Documented as an example.
3. With monotone damping and rejection of non-finite trials, most models with
   regularised kinetics need no bound.
4. `style-guide.md:448` ("use `clip_approach` when positivity is physically
   required") and `api-map.md:31` are rewritten accordingly; the `build-model`
   skill recommends the regularisation; `pitfalls.md` gains an entry "clipped
   kinetics with zero derivative below zero".

## 5. Stages

1. **Baseline harness, no library change.** Wrap `newton` to log every call made by
   the gallery pages, book notebooks, teacher solutions and plugin exemplars: site,
   size, keywords, `success`, `nit`, time, and the `residual_check` backward error of
   `x`. Store the log as the reference. Report how many calls return `success=True`
   with a backward error above `1e-8` today.
2. **Stage 1 release: changes that cannot alter results.**
   - Direct LU default, size switch removed.
   - Row equilibration.
   - `fun` at `x`, new result fields, `history`.
   - All new keywords available but off (`damping="auto"` resolves to `"none"`).

   Acceptance: harness rerun gives the same `success` and `nit`, and `x` within
   `1e-8` relative, for every call.
3. **Hard-problem test suite** in `pymrm/test/test_newton_hard.py`:
   1. trace species with `tol` in physical units (`x**2 = 1e-20`; a pellet with a
      `1e-12` species);
   2. pcec-like overshoot: `max(c, 0)**0.5` diffusion-reaction at Thiele 30 from a
      warm start; fails undamped, converges with damping and bounds;
   3. the same pellet three ways: undamped with bound 0, damped with bound 0,
      damped with the regularised rate and no bound; report iterations, success and
      error against the analytical dead-zone solution (backs section 4);
   4. Weisz-Hicks pellet in its multiplicity window (`gamma = 20`): deterministic
      branch from cold and hot guesses; `continuation` reports the fold;
   5. stiff Arrhenius pellet from a cold start;
   6. singular Jacobian (pure Neumann) gives status 5, not an exception; the double
      root `x**2 = 0` gives no false success;
   7. upwind convection with `max(c, 0)` at the round-off floor: status 1, success
      decided by the gate;
   8. invariance: iterates unchanged under row scaling by `1e6`; `scale` equal to an
      explicit change of variables;
   9. a trial point raising under `np.errstate(all="raise")` is recovered;
   10. lagged Darcy velocity via `refresh`;
   11. legacy semantics: `clip_approach` with upper bounds, a stateful `limit_step`
       callback, a direction-capping callable solver, pre-factorised `'splu'`, a
       dense Jacobian, an array `tol`, `tol=0` with `rtol`, `args`;
   12. a 2-D case with about `2e5` unknowns: LU time and memory recorded.
4. **Stage 2: flip the defaults** (`damping="monotone"` via `"auto"`,
   `backward_tol=1e-8`, `stagnation=True`) in one commit together with:
   - `pitfalls.md` P7 and its tests (`test_pitfalls.py:177-192`,
     `test_checks.py:64-70`);
   - `api-map.md`, `style-guide.md`, `model-patterns/SKILL.md` and the
     `build-model` skill, including section 4;
   - the book API pages (`pymrm-book/content/api/...`, teacher copy).

   Acceptance on the harness:
   - no call loses `success`, or each loss is listed and explained;
   - every changed `x` is listed per case (multiplicity cases may switch branch);
   - median `nfev` grows by at most one per call;
   - page run time grows by at most 20 %.
5. **Stage 3: follow-up.**
   - Add `pseudo_transient` and `continuation`.
   - Port pcec `solve_bounded` onto `newton` once a release carries it; acceptance:
     the 20-point operating map stays 20/20 with unchanged KPIs. Consider
     regularising the pcec kinetics per section 4 at the same time.
   - Migrate `clip_approach` sites to `bounds=` or regularised kinetics, one gallery
     case at a time, each rebuilt and checked.
   - Decide the future of `clip_approach` (decision 4).

## 6. Risks

- With monotone damping, cases with multiple steady states can land on a different
  branch. Only the harness comparison shows which.
- The gate turns silent false successes into reported failures. Intended, but it
  touches published gallery pages.
- The extra residual evaluation for `fun` at `x` and for the gate costs time in
  pseudo-time loops with few iterations per step.

## 7. Open decisions

1. Callback and callable-solver sites: keep them undamped under `"auto"`
   (recommended; safe for about 30 sites), or damp them as well?
   Provisional: undamped. Kept: the corpus shows these sites unchanged.
2. Gate: threshold `1e-8`, and "step test met, gate failed at `maxfev`" returns
   `success=False`? Recommended yes; this is what fixes P7. Provisional: yes.
3. `fun` at `x`: accept one extra residual evaluation per solve (up to about 30 %
   more evaluations in pseudo-time loops with 2 to 3 iterations per step)?
   Provisional: yes. Reversed after measurement (section 8): 28 % more
   evaluations over the corpus. The gate now uses the residual of the iterate
   the final step is taken from, and `fun` stays the residual before that step.
4. `clip_approach`: deprecate after the migration, or keep permanently?
   Provisional: keep, undecided until the gallery migration is done.

Not part of this plan: `pcec/solver.py:170-175` contains a dead branch (a `refresh`
check inside a block guarded by `refresh is None`); remove it when pcec is ported.

## 8. What the implementation changed in this plan

Findings from the experiments and the corpus runs that overrode the design of
section 3. Numbers in section 9.

- **Default damping is `"guarded"`, not `"monotone"`.** On 84 Weisz-Hicks
  pellets (4 parameter pairs, 7 Thiele moduli, 3 initial guesses) plain Newton
  converged in 60, `"guarded"` in 62 at the same cost, `"monotone"` in 69 at
  about twice the evaluations, and `"monotone"` lost cases plain Newton solves:
  the natural monotonicity test cannot pass the ghost of an ignition turning
  point. `"guarded"` takes the plain Newton steps wherever these are finite, so
  it can only add solutions. `"monotone"` falls back to the full step when its
  search fails ("escape"); that took it from about 50 to 69 of 84.
- **The backward error has a floor at the field scale.** Purely componentwise
  (as `residual_check`), it is about 1 in a reaction dead zone where `c` is
  tiny and correct, so the gate would never pass there. Each unknown is judged
  against `|x| + x_ref`, `x_ref` the field's largest `|x|` (or `scale`), with
  `x_ref` at least `1e-6` of the largest unknown so that a single-field 2-D state
  without a field axis is not judged column by column.
- **Row equilibration is off by default.** It changes SuperLU's pivots; a
  gallery continuation near a turning point with `tol=1e-12` then ended on a
  different branch.
- **No final evaluation.** The verdict uses the residual before the final step
  (decision 3 reversed).
- **Stagnation is opt-in and stricter.** As a default it turned 1700 corpus
  calls that failed at `maxfev` (a `tol` below the round-off floor) into
  successes, and pages built around the old outcome changed course: a
  continuation ladder (D1.1) took 86 000 instead of 35 000 solves, a page
  crashed downstream (J4.8), and a teacher solution with `rtol=1e-10` stopped
  after 9 of the 54 iterations it needs. The rule now applies only within 100
  tolerances, requires a step no smaller than the smallest of the previous
  five (slow linear convergence always makes a new smallest step), and never
  ends a solve with failure.
- **The near-solution full-step shortcut was dropped from `"monotone"`.** With
  sharp regularised kinetics (`eps = 1e-7`) full steps within `100 tol` cycled;
  instead the search is cut short near the solution (smallest damping 0.2) and
  then escapes to the full step.
- **`pseudo_transient` controls `dt` by the Newton iteration count**, not by
  the residual ratio alone: through an ignition the residual rises and the
  ratio rule shrank `dt` until the march stalled. It reached the steady state in
  18 of the 22 pellets where `"guarded"` fails; `"monotone"` solved the other
  four.
- **`continuation` rejects a corrector that jumps** (more than 20 % of `max|x|`
  from the prediction): without that check natural continuation silently
  crossed from the cold to the hot branch at the ignition fold.

