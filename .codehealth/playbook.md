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

## Refactor recipes (c2 evidence, python@1)

| id | when | when not | evidence |
|---|---|---|---|
| state-to-locals | a class talks between methods through mutable fields (class_state smell) | the field is a counter the caller reads (nfev, nfact): leave it; never hide state in nonlocal, aliases or appends | _NewtonRun 54 -> 2, _Continuation 5 -> 0, _PseudoTransient 4 -> 0; reviews R5-R8 KEEP |
| introduce-parameter-object | positional tuples read across methods; > 4 params that form one concept (settings, a correction) | a grouping that exists only for the score (coupling C9 skipped) | 4/4 accepted |
| remove-dict-as-struct | a dict with fixed literal keys used as a struct | its keys are public API; needs an interface plan (literals_removed) | newton `state` dict, S6 |
| reorder-bumps | AVOID as a step's only change: moving a plain statement so two if-blocks become adjacent lowers bumps without making the code clearer (R8 on B2) | - | 1 accepted, reviewer: metric-only |
| return-not-store | per-item results written into fields by several methods (`_Reconstructor` 14 writes): return a private NamedTuple record per item, merge in one local dict, build the arrays once at the end | an item can be recorded twice and the order of writes matters: check last-write-wins first | ibm_recon 26.0 -> 17.0, _Reconstructor 14 -> 0, 1 field-cluster step, R12 KEEP (882-case harness, 0 differences) |
| remove-overhead (neutral) | a refactor added per-iteration helper calls and NamedTuples on newton's path (cProfile shows them): win the time back with bitwise steps, measured with an interleaved in-process timer (notes/hot_timing.py), not the step gate | the faster form aliases a user-owned buffer (ff8dbed: `dy = dx` broke damping for a user solver that reuses its output; undone) | 5 neutral steps; pellet 1.032 -> 1.002 vs CB2 |
| direct-lapack | dense LU on small systems: call `dgetrf`/`dgetrs` (get_lapack_funcs) instead of scipy `lu_factor`/`lu_solve`, which add checks and a `catch_warnings` per call | non-float64 input can reach the call | factorize + 2 solves 45 -> 9 us at n=5; dense cubic 1.038 -> 0.694 vs CB2; side effect: no registry reset, a repeated warning now shows once (R12 D1) |

Step gates compare head with the campaign ref-base, so a benchmark tier reads the cumulative cost of all earlier
steps, not the step's own: a step that only removes overhead can fail bench-newton because earlier steps added it.
Measure the step against its parent (hot_timing) before deciding a re-gate (decisions 33/34).

Limits met in c2: a file above 1000 nloc gets a large_file smell that fails the gate (solve.py sits at 1000);
do not pack lines to fit. Differential harnesses (notes/fuzz_*.py, base vs head bit for bit with line coverage)
backed every solve.py step; reviewers found no difference they missed.
Bench timing: before codehealth 2847f98, wall medians sat on a 50 ms grid with a head-second bias (bench-newton
0.96-0.98 per step, 1.000 over a 5-step range); re-pinned 2026-11-18.

## c2 closing lessons (2026-11-18)

- Per-function metrics push over-fragmentation. After WO-c2-1..11, solve.py read health 9.2, but a blind review
  (no scores shown) rated it 6/10, with the Newton core at 5: about 30 methods, 10 record types and about 20 calls per
  iteration. A consolidation WO (WO-c2-12) took it to about 10 calls per iteration and still scored 6/10. Use a
  blind review as the readability check, plan for it from the start, and stop splitting before the score asks for
  it (decisions 42, 45).
- Record objects cost time in numpy-light hot loops: about 1-2 us per NamedTuple per iteration. Measure the loop
  with an interleaved in-process timer before you introduce a record type on newton's path (decision 39).
- The gate rejects a helper-only step because "reference does not execute". Commit the new helper together with its
  first caller (decision 41).
- The literals check can force two steps into one commit: when step A stops reaching a literal and step B deletes
  the scope that held it, commit A and B together. Never keep a dead literal only to satisfy the gate (decision 43).
- Benchmarks run with a cold numba cache: the gate gives every timed run a fresh NUMBA_CACHE_DIR. An optimize step
  that adds `njit(cache=True)` therefore costs about 1.6 pp cold on bench-newton (cache write) while it wins about
  1.8x warm. Decide which definition a gate criterion means before you write it, and compare direct timings under
  the same cache state (decision 46).
