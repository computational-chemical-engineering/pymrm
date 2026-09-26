---
name: conventions
description: House conventions and known pitfalls of the pymrm Python library (multiphase reactor models - finite-volume operators construct_grad, construct_div, construct_convflux_upwind, NumJac, newton). Use whenever writing, reviewing or debugging code that imports pymrm, especially boundary-condition dictionaries, array shapes, NumJac stencils, newton tolerances and outlet values.
---

# pymrm conventions

Read these before writing or reviewing pymrm code. They are short; read the
whole file that applies, not an excerpt.

| File | Read when |
|---|---|
| `references/pitfalls.md` | ALWAYS, before writing any pymrm code. Eight API behaviours that have produced wrong answers, each with the fix. |
| `references/style-guide.md` | writing a model: output format, section order, naming, array layout, class structure, residual pattern, validation |
| `references/api-map.md` | choosing a function; patterns beyond the core recipe (TVD, nested scales, 2-D, boundary unknowns, continuation) |
| `references/assembly-styles.md` | deciding how to build the Jacobian: operator sum (default), block assembly, full-residual `NumJac` |
| `references/profiles.md` | the user or project asks for efficiency, flexibility, readability or teaching value |
| `exemplars/steady_pellet.py` | a steady 1-D nonlinear boundary-value problem (sphere, `NumJac`, `newton`) |
| `exemplars/dispersion_reactor.py` | a transient 1-D convection-dispersion-reaction model (Danckwerts bc, backward Euler, constant Jacobian factorised once) |
| `exemplars/assembly_styles.py` | one two-field model in all three assembly styles, with a Jacobian cross-check |
| `exemplars/cstr_steady_states.py` | a pointwise algebraic model in compact script format (scaled unknowns, multiple steady states, two routes) |

Paths are relative to this skill's directory.

## The rules that matter most

1. Boundary dictionaries use the OUTWARD normal: `a * dc/dn + b * c = d`. Write
   the physical equation in `x` next to each dictionary and derive `a` from it.
2. Keep a field axis: one field is `(n, 1)`, never `(n,)`.
3. Face diffusivities at a jump: harmonic mean.
4. Scale unknowns to order one before `newton`; check the final residual.
5. Read boundary values with `compute_boundary_values`, never from the last cell.
6. Assemble constant operators once; use `shapes_d` when boundary values change.
   Default Jacobian: constant operator part plus `NumJac` on local terms only
   (`assembly-styles.md`).
7. Every check must be able to fail: break the model on purpose and confirm the
   checked number moves.

## Before you finish: run the checker

Run your model once through the checker and fix every finding before you report
any number:

```bash
python <this skill's directory>/scripts/check_model.py model.py
```

It runs the script with pymrm instrumented and reports NumJac on a 1-D shape
(P3, P4), newton stopping on a tolerance that is not small against the unknowns
or not converging (P7), and deprecated calls. Then check by reading your code:
every bc dictionary has its physical equation beside it and the sign of `a`
matches the outward normal at that end (P1); face diffusivities at jumps are
harmonic (P2); outlet and wall values come from `compute_boundary_values` (P8);
a stirred-volume outlet uses the sink construction (P6). Knowing the pitfalls is
not enough; check the code against them.

## API truth

The installed pymrm is the authority on signatures and behaviour. Locate it with
`python -c "import pymrm, pathlib; print(pathlib.Path(pymrm.__file__).parent)"`
and read the docstring of any function before first use. If what you read
contradicts these files, trust the code, say so in your report, and point the
user to the pymrm issue tracker.

## Copying an exemplar

Copying an exemplar copies its structure, not its validity. Its checks are for
its own physics: rebuild them for the new model (new closed form or limit, new
second route, new break row).
