# Defect classes to hunt by name

Measured across dozens of verified pymrm models (the pymrm gallery). They recur
far more often than wrong arithmetic. For each: what it looks like and the test
that exposes it.

## 1. The check that cannot fail

Two routes agree to machine precision because they share code, or because the
agreement is an algebraic identity of the discretisation. Example: a conservation
residual of 1e-11 that stayed at 1e-11 with an unconverged solve (Newton residual
18), the wrong geometry (`nu=0`) and a three-cell grid, because one
`construct_div` served both sides of the balance.

Test: break the model (flip a sign, change `nu`, use three cells, skip the solve)
and confirm the checked number moves by more than its tolerance. If it does not
move, the check is structural: it may stay as a code check, never as evidence.

## 2. A second route that is no route

The "independent" computation reads the same closed-form scalars or the same
operators as the model. Example: a reported solver result that did not change
with the geometry switched, with two cells, or with the solver deleted.

Test: break the model's solver and confirm the second route disagrees.

## 3. Coverage asserted rather than measured

A break row that returns a literal, a perturbation equal to the baseline value,
or a perturbation of a parameter the code never reads (a module constant used
instead of the configured value).

Test: for each break row, confirm the perturbed value reaches the code (print
it inside the function) and that the baseline differs from the perturbed run.

## 4. A claim that over-reads its evidence

The number is right; the sentence around it is too strong. A maximum over one
window compared with an RMS over another; "falls only 19 %" hiding an 81 % drop
in a related quantity; "negligible" for an effect that moves a headline past the
stated tolerance.

Test: for every headline sentence, name the statistic, the window and what is
held fixed, and check the comparison is like with like.

## 5. A false explanation attached to a true measurement

A real difference explained by a mechanism that is not the one in the code.
Example: an outlet difference blamed on "the upwind flux is v*c_N" when the
operator's outlet row uses the reconstructed face value.

Test: print the operator row or the intermediate quantity the explanation
names. No break row can check prose; say so.

## 6. The reference is the defect

The baseline the model is compared against is itself grid-limited, or sampled
where it should be root-found. A graded grid is not automatically accurate.

Test: refine the reference too; extrapolate; report observed orders on every
axis; root-find extrema and thresholds instead of taking the nearest sample.

## 7. Numbers that depend on the path

A reported value found at the end of a warm-start continuation chain, a
parameter sweep that reused the previous solution, or a turning point picked from
sign changes of a sampled derivative. Example: an effectiveness factor on an
ignited branch that came out 44.45 on one machine and 36.15 on another.

Test: recompute each reported value from a fixed, documented initial guess. If
more than one steady state is possible, one fixed start still finds only one:
solve from starts that bracket the branches (cold and ignited, feed and
equilibrium), check the stability of the reported state, and treat "no other
branch found" without such a search as `insufficient evidence`.

## 8. The pymrm pitfalls

Check each entry of the `conventions` skill's `pitfalls.md` explicitly: bc sign
at each end (P1), face means at property jumps (P2), `NumJac` shapes and stencils
(P3, P4), boundary values that change (P5), stirred-volume outlets (P6), newton
tolerance against unknown magnitudes and the final residual (P7), boundary
values read from faces (P8).

## 9. A misread problem statement

Two routes agree because both implement the same misreading: a stated
constant density replaced by an ideal-gas density, a rate per catalyst mass used
as a rate per volume, a boundary condition at the wrong end. Example: two
routes agreed to 1e-4 on a critical inlet pressure 16 % too high, because both
let the density vary although the statement fixed it.

Test: list every condition in the specification or the user's statement and
find the code line that implements it. A condition without a line is a finding.

## 10. Fixes that bring new defects

When verifying a second time, after a fix, look hardest at what the builder
ADDED in the fix, not only at the findings it addressed. Correct repairs of real
findings have repeatedly introduced a new defect of classes 1, 6 or 7.
