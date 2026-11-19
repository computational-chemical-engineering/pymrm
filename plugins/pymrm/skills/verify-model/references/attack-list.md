# Defect classes to hunt by name

Found repeatedly while verifying the pymrm gallery's pages
(<https://github.com/computational-chemical-engineering/pymrm-gallery>). They recur
far more often than wrong arithmetic. For each: what it looks like, the test that
exposes it, and sharper attacks with a measured example (gallery catalogue ID;
"(gl N)" is row N of the gallery's `docs/refresh-learnings.md`).

## 1. The check that cannot fail

Two routes agree to machine precision because they share code, or because the
agreement is an algebraic identity of the discretisation. Example: a conservation
residual of 1e-11 that stayed at 1e-11 with an unconverged solve (Newton residual
18), the wrong geometry (`nu=0`) and a three-cell grid, because one
`construct_div` served both sides of the balance.

Test: break the model (flip a sign, change `nu`, use three cells, skip the solve)
and confirm the checked number moves by more than its tolerance. If it does not
move, the check is structural: it may stay as a code check, never as evidence.

- Surface flux and volume average of a linear pellet are one telescoped sum; the gap is
  the solve residual (`B1.2`: 8.1e-10). Its break row is a two-point read (gl 93).
- Classify by what the quantity is: an integrator is exact on a constant rate (`B2.3`:
  2.7e-15, stated bound 1e-7); cells that cannot affect a quantity give orders -6.5, 6.8
  (`A3.7`) (gl 129, 237).
- An identity of full Newton steps fails after a `bounds`-shortened step (`B1.6`: 1 of
  324) (gl 116).
- "Structural" is often a missing row: a guard computed in the rows' helper moved 2.6e4
  to 0 (`D1.1`) (gl 54, 205).

## 2. A second route that is no route

The "independent" computation reads the same closed-form scalars or the same
operators as the model. Example: a reported solver result that did not change
with the geometry switched, with two cells, or with the solver deleted.

Test: break the model's solver and confirm the second route disagrees.

- List what is shared: two roots on one spline agreed to 1.2e-14, below their bisection
  resolution (`E1.2`); "no shared code" shared the end condition (`E1.1`) (gl 150, 153).
- Two estimates from the same rows are two weightings, not two pieces of evidence (gl
  204).
- A solver stalling on a broken model is no detector: with the exact Jacobian all three
  "alarmed" defects converged (`A2.1`) (gl 61).

## 3. Coverage asserted rather than measured

A break row that returns a literal, a perturbation equal to the baseline value,
or a perturbation of a parameter the code never reads (a module constant used
instead of the configured value).

Test: for each break row, confirm the perturbed value reaches the code (print
it inside the function) and that the baseline differs from the perturbed run.

- One row per claim, sized to fail its tolerance: `B1.7` ran rows on 8 of 53 mapped
  metrics; under 5 % a `k2 + 1 %` row is invisible (`J4.6`) (gl 7, 119).
- Inject a wrong parameter everywhere it enters, post-processing included: `D2.1` 12.4 %
  against 22.66 %; a geometry row gave 1.032 for 4.5 (`E1.2`) (gl 28, 154, 191).
- A defect's effect is two solves on one grid, not each against the reference (`A2.8`:
  5.9e-7 grid error, 2.5e-16 effect) (gl 70).
- Rows that inject nothing: a face-mean row on a uniform `D` (`A3.2`); one bad cell
  absorbed by a mask (`A1.5`) (gl 86, 158).
- Add the row where an approximation is exact (`A3.1`: a wrong resistance sum missed it
  by 76 %); a null predictor and a "none match" count each need a row (gl 78, 206, 219).

## 4. A claim that over-reads its evidence

The number is right; the sentence around it is too strong. A maximum over one
window compared with an RMS over another; "falls only 19 %" hiding an 81 % drop
in a related quantity; "negligible" for an effect that moves a headline past the
stated tolerance.

Test: for every headline sentence, name the statistic, the window and what is
held fixed, and check the comparison is like with like.

- "Every", "never", "whole range": evaluate every case and swept value. "Every voidage"
  held at 4 of 5 (`A1.8`); "at most 15 %" was 138 % on the data (`F1.3`) (gl 94, 142,
  143, 170, 190, 195, 222, 232).
- Count by code with a margin, gaps in half-units of the last printed digit (`H1.9`:
  2.5); "outside every dataset" was inside for 5 of 9 (`H1.8`) (gl 44, 45, 160, 162,
  172, 194, 211, 214, 218).
- Rows related by a symmetry count one case twice (`A2.5`: 12 rows, 8 beds) (gl 65).
- Name the denominator and the reading: a factor below a bound divides by it (`I1.3`:
  2.9, not 2.0); a mean of ratios is not a ratio of means (`F1.3`) (gl 52, 73, 106, 171,
  215, 217).
- Compare a correction with its own correlation without it, and root-find where a
  comparison flips (`A1.1`: only up to x = 378) (gl 144, 145).
- A sensitivity goes through the coupled system with what is fixed named (`G1.7`: 44 %
  quoted, 21 to 41 % coupled) (gl 29, 168).
- A conservatism factor divides by the root-found point (`A3.3`: 62, not 53); split data
  where the abscissa certifies (gl 91, 179).

## 5. A false explanation attached to a true measurement

A real difference explained by a mechanism that is not the one in the code.
Example: an outlet difference blamed on "the upwind flux is v*c_N" when the
operator's outlet row uses the reconstructed face value.

Test: print the operator row or the intermediate quantity the explanation
names. No break row can check prose; say so.

- Remove the named cause and see if the effect stays: smoothing `sqrt(y)` left the stall
  at 9.44e-6 (`H1.4`) (gl 37, 79, 123, 188).
- Check the read before the physics: an O(h) closure was the outlet read (`H1.4`: 3.5e-3
  to 2.0e-14); a "+3.8 % effect" a two-point Dirichlet read (`A3.3`) (gl 35, 58, 87,
  166).
- Name the right part: the limiter, not the integrator, broke an identity (`A2.5`:
  2.9e-15 upwind, 2.75e-6 van Leer); a fit window reaching the outlet (`A3.15`) (gl 69,
  77).
- Truncation shrinks with the step, round-off grows: retry a difference quotient on a
  series (`B1.4`) (gl 109).
- Newton reaches unstable states as readily as stable ones (`B1.1`); a matched constant
  is not the assumption holding (`J1.5`) (gl 18, 180).
- A stated failure mode or necessity needs the code that shows it (`D1.1`: "123 K", 0.52
  K in every build) (gl 53, 60, 212).

## 6. The reference is the defect

The baseline the model is compared against is itself grid-limited, or sampled
where it should be root-found. A graded grid is not automatically accurate.

Test: refine the reference too; extrapolate; report observed orders on every
axis; root-find extrema and thresholds instead of taking the nearest sample.

- Check a reference at two tolerances, `atol` scaled to the state (`H1.4`: moved
  3.2e-7); `brentq` on an adaptive integrator inherits its noise (`B1.7`: order 1.953)
  (gl 36, 117).
- Measure series truncation at the largest parameter used (`B1.2`: 3.0e-5 there, 1e-9
  where validated) (gl 98).
- Closed forms cancel: compare with 40-digit `mpmath` (`B3.3`); check exported tables
  row by row and limits at a large argument (`B1.3`: 41.4 % was 100 %) (gl 99, 100,
  137).
- Read the reference where the solve is, not at the first cell (`B1.6`: offset 3.944e-5,
  error 1.98e-6) (gl 111, 114).
- Richardson removes discretisation, not estimator bias (`A2.3`: converged cleanly to
  -0.416 %) (gl 63, 64).
- Measure the reference's error on every rung (`B2.3`: slope 1.009, 1.000 with a finer
  one) (gl 127, 209).
- Test a far field by nested domain extension (`E1.2`: Robin 9.6e-12, Dirichlet 1.9e-5)
  (gl 155).

## 7. Numbers that depend on the path or the machine

A reported value found at the end of a warm-start continuation chain, a
parameter sweep that reused the previous solution, or a turning point picked from
sign changes of a sampled derivative. Example: an effectiveness factor on an
ignited branch that came out 44.45 on one machine and 36.15 on another.

Test: recompute each reported value from a documented start. If more than one
steady state is possible, one fixed start still finds only one: solve from starts
that bracket the branches (cold and ignited, feed and equilibrium), check the
stability of the reported state, and treat "no other branch found" without such
a search as `insufficient evidence`.

- Near a threshold a cold start lands on either branch by round-off (`D1.1`: 990.9,
  411.5, failure, 992.1 C). Reproduce it along the branch from one cold solve at a safe value, and along a
  second path (halved steps, or from the other end); name the branch by a state
  criterion; a gap beyond the solver tolerance is path dependence (gl 8).
- Rerun under `OPENBLAS_CORETYPE=Sandybridge` and `Nehalem` (OpenBLAS only; it reproduced four of four
  runner failures); a pre-asymptotic order moves with the environment (`J1.10`: 1.79,
  2.11) (gl 3, 9).
- Never state, divide by, count or classify by value a round-off quantity (`J4.8`: 7e-12
  was 0.0 on another kernel) (gl 1, 2, 6, 40).
- Round-off only if tightening every tolerance leaves it (`A3.2`: 1.27e-12 was `quad`'s
  `epsabs`); thresholds sit between round-off and the smallest catch (gl 86, 223).
- An optimiser's stop on an ill-conditioned valley is noise (`B2.3`: 9.5e-9 to 9.1e-6
  under a 1e-15 change): use `jac="3-point"`, a large `diff_step`, a profile (gl 11).

## 8. The pymrm pitfalls

Check each entry of the `conventions` skill's `pitfalls.md` explicitly: bc sign
at each end (P1), face means at property jumps (P2), `NumJac` shapes and stencils
(P3, P4), boundary values that change (P5), stirred-volume outlets (P6), newton
tolerance and the final residual (P7), boundary values read from faces (P8),
clipped kinetics (P9), deferred-correction stalls of steady TVD (P10).

- P8 is wider than outlets: upwind cells hold the downstream face value (`E1.1`:
  0.258305 for 0.258472); a boundary-layer first cell; exact cell volumes for means
  (`J1.5`) (gl 88, 148, 183, 199).
- P7: accept a step on `success`, not an own threshold (`J3.5`: 18 rejected steps had
  `success=True`); P9: a silenced `MatrixRankWarning` was active clips (`J3.4`) (gl 198,
  208).

## 9. A misread problem statement

Two routes agree because both implement the same misreading: a stated
constant density replaced by an ideal-gas density, a rate per catalyst mass used
as a rate per volume, a boundary condition at the wrong end. Example: two
routes agreed to 1e-4 on a critical inlet pressure 16 % too high, because both
let the density vary although the statement fixed it.

Test: list every condition in the specification or the user's statement and
find the code line that implements it. A condition without a line is a finding.

- Run each reuse or variation against a closed form: changing only `nu` missed the slab
  by 0.67 (`G1.8`) (gl 30).
- A law with two readings: no balance identity or single number separates them (`F2.3`);
  a reference group is what the source says it is (`F1.4`) (gl 176, 185).

## 10. Fixes that bring new defects

When verifying a second time, after a fix, look hardest at what the builder
ADDED in the fix, not only at the findings it addressed. Correct repairs of real
findings have repeatedly introduced a new defect of classes 1, 6 or 7.

- A list of corrections is a claim about the old version: check each against it; a
  sharper true value is not a correction (`A3.3`) (gl 92, 97).

## 11. Read off a sample instead of solved for

A threshold, extremum, crossing or limit taken from a grid node, sweep point or
interpolant reports the sampling. Test: root-find on the model
(`pymrm.checks.find_roots`, a derivative root, bounded maximisation after a scan).

- Root-find probe states and classes: interpolation gave 2.9e-3, roots 1.4e-5 (`B1.1`)
  (gl 17, 107).
- Scan, then refine extremes: a bounded minimiser is local (`B1.3`: 0.098 was 3.05e-6);
  a grid argmax on a kinked crest wanders (`B3.1`) (gl 96, 105, 131, 202).
- An extremum at a domain or grid end means "at or beyond" (`C2.10`: 6.65 m, not the 3 m
  exit) (gl 22, 182, 189).
- Bisect well below the quoted margin (`C2.10`: +0.0078 from a 0.0156 bracket); an onset
  is the cusp (`B1.4`) (gl 21, 108).
- Interpolation reads: `np.interp` before the first centre returns the first cell, so a
  profile error is the half-cell gap (`B3.2`: 2.4e-2, 2.9e-5 at centres); observation
  times off the step grid add a second-order read-off error to an observed order;
  `t += dt` with a `t >= t0` filter drops the window edge (gl 124, 125, 126, 133, 135).
- Band edges are roots (`A3.1`: [1.14, 1.24] was [1.126, 1.244]); a capped root scan
  hides roots; a flat noisy minimum resolves to `sqrt(2 dF/F'')` (gl 81, 138, 169).
- A refined peak at a fixed step is still that step's value, with the integrator's own
  error (`I1.2`: 3.70 %, 3.36 % extrapolated); a node count is a grid estimate of an
  area, off in the last printed digit (`A3.7`) (gl 48, 238).

## 12. An error budget that measures the wrong axis

An order or accuracy that mixes axes, rungs or cases. Test: orders from
successive differences of three levels (`pymrm.checks.observed_orders`), per
case, per axis, on the published statistic.

- Against the finest run or the exact value a ladder carries that error (`H1.9`: 2.39
  was 2.09; `A3.15`: 2.12 was 2.01) (gl 43, 75, 134).
- Report per case, at a second parameter (`B1.1`: mean 2.35 was 2.96, 2.06, 2.03) (gl
  19, 122).
- Axes cancel on 2-D grids (`A3.15`: 5.5e-6 from +5.9e-5 and -5.3e-5): report the larger
  (gl 74).
- Split step and grid before naming either (`J1.5`: 97 % was step; `A3.3`: step-only
  Richardson left grid error) (gl 82, 83, 89, 181).
- Refine the phase, layer and statistic that carry the error (`J3.4`: 0.93, 1.002 with
  split steps; `A3.6`: 8.4e-7 was 8.4e-11) (gl 66, 201, 226, 231).

## 13. A solver's stop read as a result

A value held when a solver stopped, failed or hit a bound, published as physics.
Test: check every status (`newton` `success`, `solve_ivp` `status`, optimiser
bounds) for each reported value.

- `solve_ivp` `status = -1` returns a partial trajectory (`D2.1`: "nothing runs away"
  was 1.45 K at the stop) (gl 26).
- A bisection predicate counting failure as "above" (`D2.2`: a 1937 K iterate); bracket
  ends where an inner loop diverges (gl 25, 34).
- A defect may leave no solution: record nan, prove it by a second route (`A2.8`:
  the stopped iterate of an unchecked loop was published as "28984") (gl 72).
- A minimiser on its bound (`C1.3`: rms 8.91 at k = 3000, 2.78 at the minimum);
  `success` barely below `tol` (gl 102, 193).
- `newton` raises `NewtonBreakdown` on a singular Jacobian rather than return
  `success=False` (`B3.3`: 13 of 20) (gl 139).

## 14. A statistical verdict without a standard error

A gap, bias or fitted value called large, small or inside a band without its
standard error. Test: compute it (paired, or from the fit covariance with its
degrees of freedom) and state the verdict in standard errors.

- A paired difference of deviations (`A2.6`: 2.7 points, 0.89 SE); a mean bias or median
  near 1 (`A1.7`: SE 9.6 %) (gl 177, 216, 228).
- A fitted intercept, not replicate scatter (`A3.5`: "20 sigma" was 3.08 SE); a rounding
  floor is not scatter; add confounds (`B2.1`) (gl 161, 203, 204, 235).

## 15. Numbers no code produces

A number, count or label typed into prose, a README or metadata. Test: trace
each to the line that prints it; recompute ratios from the loaded data.

- README and metadata against the run (`I1.2`: closures 3.7e-8 against 5.3e-9) (gl 23,
  49).
- Counts and multipliers in prose (`F1.3`: "ten times" was 7.8); figure reads point by
  point (gl 55, 174, 178).
