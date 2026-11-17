# pymrm code-health playbook

Read after the seed playbook of the refactor-loop skill (cce-code-health). Evidence from campaign c1
(2026-11-16, src/pymrm, debt 1282.8 -> 80.0, display health 5.82 -> 9.57, brain functions 8 -> 0;
`codehealth stats --campaign c1`: 105 gated attempts). Run log: `.codehealth/runs/c1.jsonl`.

## Rules for new code

1. Keep every function under cognitive 15, 70 lines of code, nesting 3, and 5 parameters; a public
   function may take more parameters, a private helper may not: group inputs that form one concept
   into a private NamedTuple or dataclass (`_FieldGeometry`, `_ReconOptions`), never a grab-bag.
2. Boundary-condition and geometry plumbing lives once, in `helpers.py` (`_field_geometry`,
   `_wall_alphas`, `_wall_factor`, `_unwrap_bc_side`); operators, convection and interpolation call
   it. A lower-level module never imports a higher one (operators must not import convect).
3. A solver loop with shared state is a private class with phase methods (`_NewtonRun`,
   `_Continuation`, `_PseudoTransient`); keep per-iteration values in locals and return values,
   declare the object's state once and use slots: attribute-heavy loops cost about 2 % per solve on
   small problems (R2).
4. On per-call hot paths (residual evaluation, Newton iteration, ConvFluxTVD.__call__,
   compute_boundary_values, apply_ibm): no new array copies, no per-element Python loops, no
   repeated broadcasting; prefer ndarray methods (`a.all()`, `a.max()`, `a.diagonal()`) over the
   `np.*` function wrappers (a few microseconds each, measurable on small systems).
5. A `warnings.warn` reached through an extra private call layer needs `stacklevel + 1`; the
   bitwise test gate cannot see a warning attributed to the wrong frame (R3).
6. Never rename or remove a private name that `test/` or `plugins/` reference (tests monkeypatch
   `pymrm.solve._continuation_predictions`); `checks.py` has a byte copy in `plugins/`.
7. String literals in error messages belong to the function that raises; a shared helper may carry
   them only when every original caller still reaches it.

## Refactor recipes (c1 evidence)

| id | when | when not | evidence |
|---|---|---|---|
| extract-phase | long function with sequential phases (validation, geometry, assembly, output) | phases share many mutable locals; the helper would need > 5 inputs (do parameter-object first) | 50 tries, 41 accepted (82 %), median drop 9.3, 10 files |
| closures-to-class | function whose nested closures share state through nonlocal or a state dict (newton) | the loop is hot and the closure locals become dozens of attributes: hoist loop values to locals, declare state with slots | 1 try, 136.1 debt drop (68 % relocated, followed by phase steps) |
| consolidate-duplicate | identical blocks in 2+ functions or modules (BC per-wall code) | blocks differ in floating-point order or dtype handling (`+ np.zeros` vs `broadcast_to`); a NamedTuple that one path fills late | 15 tries, 7 accepted + 8 neutral, 5 files |
| parameter-object | 6+ inputs forming one concept passed through private helpers | the fields are produced in different phases | 9 tries, 5 accepted, 5 files |
| speed-overhead (neutral) | profile shows Python-level overhead on a hot path: redundant copies (`csc_array(jac)` of a canonical csc), `np.*` wrappers in per-iteration reductions | the faster form changes floating-point results or order | newton: pellet -3.3 %, dense -8 % per solve, bitwise |

Gate failures to expect: wall-time benchmarks under machine load (use metric "both"; re-gate once only
when the failing benchmark does not run the changed code), literals (a dropped string literal), and
coverage (numba-compiled functions are invisible to coverage: leave them).
