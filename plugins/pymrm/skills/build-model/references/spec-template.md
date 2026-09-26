# Specification template (`spec.md`)

The specification is the contract. The implementation is judged against it, and
the verifier reads it without the conversation that produced it, so it must
stand alone. Keep it short; put derivations in an appendix.

```markdown
# Specification: <short name>
Status: DRAFT | APPROVED by <user> on <date>
Mode: full
Brief: brief.md   Scoping: scoping.md

## 1. Question
<the decision, and the outputs that answer it, with required accuracy>

## 2. Model
Structure codes: <S..>
Geometry and coordinates: <domain, nu, dimensionless or dimensional>
State layout: <e.g. c[(n_z, n_c)], fields and their order>
Unknown scaling: <reference value per field, so newton sees order-one unknowns>

Equations (one per balance, in the form the code will implement):
<...>

Boundary conditions (one line each: physical equation, then the pymrm dict):
- z = 0, species: v c - D dc/dz = v c_in   ->  {"a": D, "b": v, "d": v c_in}   (outward normal -z)
- z = L, species: dc/dz = 0                ->  {"a": 1, "b": 0, "d": 0}
<...>

Initial condition (transient only): <...>

## 3. Parameters
| Symbol | Value | Unit | Label | Source or reasoning |
|---|---|---|---|---|
| D_eff | 1.0e-6 | m2 s-1 | correlation: <name, valid for ...> | <...> |
| k | ... | ... | assumed | order of magnitude from <...>; varied in sensitivity run S1 |
Labels: user, literature: <reference>, correlation: <name and range>, assumed.

## 4. Assumptions
<numbered; each with the reason it is acceptable, pointing to scoping.md>

## 5. Numerics
Grid(s): <n per axis, clustering>   Time stepping: <scheme, dt>
Solver: <newton options, continuation if any, how multiplicity is handled>
Jacobian assembly: <operator sum | block assembly | full-residual NumJac | hybrid, and why>
Profile: <from brief.md, and the choices it drove>

## 6. Validation plan (numbered assertions)
Each: what is compared, against what, tolerance, why the tolerance is right,
and how we know the check can fail.
A1. <exact solution or limit>: <quantity> within <tol> of <closed form>.
    Can fail because: <e.g. flipping the bc sign moves it by 20 %>.
A2. <second independent route for headline X>: <method>, shares no operators
    with the model; agreement within <tol>.
A3. Refinement: <quantity> on n, 2n, 4n (and dt, dt/2, dt/4); observed order
    within <range>; reported value's discretisation error below <tol>.
A4. <conservation or identity>: state whether it is structural (exact by
    construction). A structural identity is a code check, not evidence.
A5. <sensitivity>: headline X for each `assumed` parameter at 0.5x and 2x.

## 7. Outputs
<table or figure list; every number printed by code>

## 8. Stopping rule
<from brief.md: what counts as done; one fix round after verification>

## 9. What this model will not establish
<e.g. no validation against plant data; kinetics outside 500-600 K untested>
```
