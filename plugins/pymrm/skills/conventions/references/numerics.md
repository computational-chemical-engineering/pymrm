# Numerical practice around a pymrm model

Mistakes seen repeatedly in coding-agent runs on pymrm models (2026 evaluations
of this plugin) that are not pymrm API pitfalls but cost the most debugging
turns. Each rule is short; follow it the first time.

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
