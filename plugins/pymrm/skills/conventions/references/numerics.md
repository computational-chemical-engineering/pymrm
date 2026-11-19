# Numerical practice around a pymrm model

Mistakes seen repeatedly in coding-agent runs on pymrm models (2026 evaluations
of this plugin) that are not pymrm API pitfalls but cost the most debugging
turns. Each rule is short; follow it the first time. Rules marked "(gl N)" come
from refreshing the published pymrm-gallery pages: row N of the gallery's
`docs/refresh-learnings.md`, with the measured example named by catalogue ID.

## Use the check tools instead of writing your own

`pymrm.checks` (pymrm after 2.3.1) has four deterministic tools. On an older
pymrm, the same file ships with this skill as `scripts/pymrm_checks.py`:

```python
try:
    from pymrm.checks import check_jacobian, observed_orders, find_roots, residual_check
except ImportError:  # pymrm 2.3.1 or older: use the copy in this skill's scripts/ folder
    import sys; sys.path.insert(0, "<this skill's directory>/scripts")
    from pymrm_checks import check_jacobian, observed_orders, find_roots, residual_check
```

- `check_jacobian(fun, x)`: your Jacobian against central differences, at a
  state away from the solution.
- `observed_orders(solve, ns)`: refinement study with observed orders and a
  Richardson error estimate; `ns` must have a constant ratio.
- `find_roots(f, lo, hi)`: every sign change in a range, failed evaluations
  reported separately; use it for thresholds (runaway, ignition, critical inlet).
- `residual_check(fun, x)`: scale-free backward error of a solution.

## Judge the final residual relative to the size of the terms

`newton` stops on the step (pitfalls P7), so check the residual yourself, but
not against an absolute number. A threshold of 1e-10 is right for a
dimensionless equation and wrong for a balance in mol m-3 s-1, where the terms
are 1e3 and round-off alone leaves 1e-10. Compare with the scale of the terms:

`residual_check(fun, x)` does this for you: it divides each residual by the
size of the terms that make it up (a componentwise backward error), so it gives
the same verdict whatever the scaling of rows and unknowns (not across offset
units such as degrees Celsius against kelvin). Or scale the equations to order one (divide each
balance by a reference rate) and then use an absolute threshold. Never raise "not converged"
on a solve that reports convergence without first printing both numbers.

## Keep tolerances above the round-off floor

- `newton`'s `tol` below `cond * eps` of the unknowns never stops: the steps
  jitter at the floor. `A2.4` (Pe = 0.1, 1500 cells): `tol=1e-12` fails after
  100 iterations, `tol=1e-10` converges in 2; `B1.3` on 800 cells hits `maxfev`
  at `tol=1e-13` with a backward error of 8e-17. Use `tol` above cond*eps times the
  largest unknown, or `stagnation=True`, and require `success` (gl 13, 104).
- The floor depends on how the residual is written. Entries of `D/dz^2` = 1e6
  on order-one values leave 4e-10 on `A2.1`; the TVD residual as a correction
  to the upwind solution, `jac_up @ (c - c_up) + div @ (F_tvd(c) - F_up(c))`,
  lets Newton land on the deferred-correction state to 1e-15 with a shared LU
  factorisation, on every BLAS kernel, when the upwind problem is linear (gl 59).
- A value below the solver tolerance is a bound: `F3.2` printed -9.3e-58 and
  +3.5e-28 by two routes at `tol=1e-10`. The location of a flat extremum
  amplifies the tolerance (values 1e-9, argmax 5e-6): hold it to a value
  tolerance (gl 270, 271).
- A finite-volume balance and the `compute_boundary_values` flux agree to the
  volume-weighted solve residual, which grows with stretched cells (`A3.3`:
  4.6e-10, rows of 4e8); bound the gap from it. A two-point wall gradient is
  first order (`B3.1`: 1.2e-2 against 2e-6) (gl 88, 93, 130).
- Rerun under `OPENBLAS_CORETYPE=Sandybridge` and `Nehalem` to see another
  machine's round-off on yours; it reproduced the four gallery CI failures tried (`F3.2`, `J4.1`, `J4.6`, `J5.3`)
  where NumPy uses OpenBLAS (gl 3).

## Bracket before you root-find

`find_roots` does the following for you. `brentq` needs a sign change. Before
calling it yourself, evaluate the function on a
coarse grid over the physically possible range, pick the interval where the sign
changes, and handle the cases explicitly: no sign change (report that the
threshold is outside the range, do not widen the bracket blindly) or several
(multiplicity; report all). For thresholds found on a model (runaway limits,
ignition, critical inlet values), the function is often only defined where the
model converges: make it return a clear value or raise a distinct error when the
solve fails, so a failed solve is never mistaken for a sign.

- A predicate that counts `success=False` as "above" reads failures as physics
  (`D2.2` took a 1937 K non-converged iterate as a hot spot). An inner
  fixed-point iteration must check its own convergence: a bracket end near a
  singular limit can sit where it diverges (`G1.1`) (gl 25, 34).
- Near multiple steady states a cold start per evaluation lets round-off pick
  the branch (`D1.1`: neighbouring values gave 990.9 C, 411.5 C, failure,
  992.1 C). Follow the branch: one cold solve at a safe value, then warm-start
  every point and midpoint from the converged state at the lower end (gl 8).
- A value where a functional on a single-valued branch reaches a target is a
  scalar root on independent solves, not a fold (gl 25). The onset of
  multiplicity is the cusp, the root of the largest slope along the branch, not
  a fold-depth threshold (`B1.4`: 4.86 against 4.91) (gl 108).
- Bisect well below a margin you quote, and compare thresholds only from the
  same grid and tolerance (`C2.10`: margin 0.0078, bracket 0.0156) (gl 21).
- Scan wide on a log grid: capped at 50, `E2.1` found 4 of its 6 roots. A pole
  gives a spurious sign change; multiply by the denominator (`F1.3`) (gl 169, 173).

## Locate extremes and crossings instead of reading samples

- A sampled extreme or fold is known to the sample spacing, and interpolating
  two curves at a probe measures the interpolation (`B1.1`: 2.9e-3 against
  1.4e-5 root-found). Root-find probe states and maximise extremes directly; a
  grid argmax on a kinked crest wanders with the grid (`B3.1`) (gl 17, 131, 220).
- Classify states near a threshold by root-finding them, not by interpolating a
  traced branch (`B1.4`: 14 of 2349 changed class) (gl 107).
- Match a reference at the same point: `B1.6` compared at the first cell, half
  a cell from the centre, and published 3.9e-5 for an error of 2.0e-6 (gl 111).
- A refined crossing at a fixed step is still that step's value: extrapolate in
  the step before comparing (`I1.2`, `I1.3`) (gl 48, 51).
- Read a march at step times: interpolation between steps made Heun look third
  order on `B2.3`. Land one implicit step on the target instead (`J3.1`)
  (gl 126, 288).

## When Newton does not converge

- Take the guess from the limit the equations reduce to: on `H1.9` a guess
  linear in `ln Theta` failed under every damping, one linear in `Theta` (exact
  in the Henry limit) converged with `damping="monotone"` in 7 to 9 (gl 42).
- Test a named cause of a stall by removing it: smoothing the `sqrt(y)` blamed
  on `H1.4` left the edge in place, and the iterates past it had residuals of
  1e2, a solver limit. Keep operating points well below such an edge (gl 37).
- For a sweep from an easy state, `pseudo_transient` needs no path: on `B1.6`
  `continuation` stopped at folds in 17 of 324 cases, `pseudo_transient` in
  none, given `bounds` theta >= 0.2 against overflow of `exp(gamma/theta)` (gl 112).

## Steady and transient TVD: which route

P10 gives the default (`ConvFluxTVD`, `damping="monotone"`). Measured:
steady 1-D Newton is about five times faster than van Leer deferred correction
(`A2.8`, same solution to 4.6e-14) (gl 71). In a 1-D Crank-Nicolson step both
solve the same equations and deferred correction is twice as fast (`A2.5`, gap
2e-9) (gl 67). On large steady 2-D grids the Jacobian LU has about four times
the fill, and deferred correction on one upwind LU is 4 to 9 times faster, but
only under-relaxed (van Leer, omega = 0.6: 23 to 49 sweeps; omega = 1 does not converge).
Use it there with Newton as the check on a coarser grid (`A3.15`, `I1.3`) (gl 50, 76).

## Refinement studies

- Split the error by axis before naming its cause: the time ladder on a fixed
  grid, the grid ladder with time exact (eigendecomposition of the constant
  operator) or cancelled (outlet values at fixed record times, one step count).
  `J1.5`: 97 % of a "steep profile" error was the step (gl 122, 181).
- On a 2-D grid one grid's error can be the two axes cancelling (`A3.15`:
  5.5e-6 from +5.9e-5 and -5.3e-5). Measure each axis on its own ladder, check
  they add on a third grid, report the larger (gl 74).
- After Richardson in the step the grid error dominates (`A3.2`, `A3.3`):
  extrapolate both axes, and the reference too when its error falls slower
  than the one it judges (`J3.5`) (gl 83, 89, 209).
- Nest ladders: scaling `dt0` and `dt_max` of a growing schedule does not
  refine its growth phase (`J3.4`: 0.93); splitting each base step into m
  steps does (1.002) and keeps observations on step times (gl 201).
- Report orders per case, as a range with the ladder. A thin reacting layer
  gives 2 to 3 (`B1.2` Lambda 20: 2.72 to 2.78); an average hides it (`B2.2`:
  1.93 was 2.02, 2.08, 1.69); a rate singular at the inlet keeps upwind below 1
  (`C1.3`: 0.90 to 0.93). Ratio 2.29 at 1.33x refinement is order 2.9, not 2
  (gl 95, 122, 197).
- Check the error is monotone on the ladder: an exponential stretch changed its
  sign between 400 and 800 cells on `A3.6`; geometric cells, natural for `1/r`,
  gave 2.05, 2.03, 2.01 (gl 230).
- First- and second-order errors of opposite sign approach order 1 from below
  (`A3.5`: 0.875 to 0.986); `(8*q(h/4) - 6*q(h/2) + q(h))/3` removes both
  (3e-10 left). Say whether an order is against a closed form or from
  successive differences: they differ (`J4.4`) (gl 236, 299).
- Richardson removes discretisation error, not bias: `A2.3`'s window opened
  before start-up decayed and converged to -0.42 %; a fit ending at an outlet
  picks up an effect that halves with h (`A3.15`: 0.98 % against 0.009 %).
  Test a "floor" against an exact reference before naming it (`A3.1`)
  (gl 63, 77, 79).
- Extrapolate the step of a centred difference in a family parameter before
  quoting a trend (`D2.1`: -10.28 % at fixed h, -9.76 % extrapolated) (gl 27).
- A small derived quantity needs its own ladder (`A2.5`: a lag with the wrong
  sign on 500 cells), and a sweep is checked at every swept point (`B1.2`:
  3.3 % off at the largest moduli until graded) (gl 66, 94).
- Prove a far-field claim by nested domain extension (`E1.2`: Robin far field
  moved 1e-11 from 20 R to 400 R, Dirichlet at 20 R 1.9e-5) (gl 155).

## Make the reference better than what it checks

- Scale `atol` with the state and check at a second tolerance: Radau at
  `atol=1e-14` on flows of 8e-7 mol/s moved `H1.4`'s outlet by 3.2e-7 (gl 36).
- A root or Jacobian taken from an adaptive integration inherits its non-smooth
  error: a `brentq` reference on DOP853 at 1e-10 gave `B1.7` a false order 1.95
  (2.000 at 1e-12) (gl 117). For `least_squares` on an ODE use
  `jac="3-point"` with `diff_step` well above the integrator tolerance; `B2.3`'s
  default columns were off by 7e-2 (gl 11).
- Closed forms lose digits to cancellation: use `expm1`, `log1p` or a series,
  root-find quantities spanning decades in log or logit (`B1.3`), and compare
  with `mpmath` at 40 digits before blaming a floor (`B3.3`) (gl 100, 137).
- Measure a series for truncation at the largest parameter it is used at
  (`B1.2`: 1e-9 at Lambda 1, 3e-5 at 50) (gl 98).
- A linear multi-stream plug-flow model is solved exactly in the eigenmodes of
  `V^-1 S`, each normalised at its larger end (condition 28 against 7.7e59,
  `E2.1`) (gl 167).

## Marching and quadrature

- Check `solve_ivp`'s `status` as you check `newton`'s `success`: `status=-1`
  returns a partial trajectory. Where a rate grows without bound, integrate in
  the consumed reactant (`dT/dp`, `dz/dp`) (`D2.1`) (gl 26).
- Measure before choosing an integrator or a step map (`A3.2`: BE extrapolated,
  BDF2 and CN within a factor 3; steps in `sqrt(zeta)` 2 to 5 times worse) and
  before choosing a centre or face read of upwind (`C2.1`: centre 2.7 times
  closer) (gl 84, 140).
- Spline the smooth product `r * alpha**p`, not a singular rate, and integrate
  in `u = sqrt(alpha)` (`C1.3`: quadrature gap 2.3e-6 to 2.3e-10) (gl 196).

## Keep scripts where they can import the model

Write driver and check scripts in the model's directory, or run them with that
directory on `PYTHONPATH` (`PYTHONPATH=/path/to/model python check.py`). A
script in `$TMPDIR` or a subfolder does not see the model module otherwise.

## Check optional tools before relying on them

pymrm needs numpy and scipy; anything else may be missing (`sympy`, `nbformat`,
`jupyter`, `matplotlib`, `/usr/bin/time`). Check once
(`python -c "import sympy"`), and if it is absent derive by hand, use a
numerical route, or tell the user what to install. Do not build a plan on a
package you have not checked.

## Tolerances that SciPy refuses

`brentq` raises an error for `rtol` below 4 machine epsilons (8.9e-16); use
`xtol` for the absolute accuracy you need and leave `rtol` at its default.
`solve_ivp` raises `rtol` below about 2.2e-14 to that value with a warning; use
`rtol=1e-12` with a small `atol` for a reference solution. `quad` warns when it
cannot meet `epsabs` and `epsrel` near 1e-13; raise `limit` or split the
interval rather than tightening further. A reference solution needs to be a few
digits better than the quantity it checks, not at machine precision.

## Build sparse inputs from float arrays

Pass numeric `numpy` arrays (float64) to `construct_coefficient_matrix` and bc
dictionaries. A list that mixes arrays and scalars becomes an object array and
scipy.sparse rejects it ("does not support dtype object").
