---
name: model-patterns
description: Patterns for coupled pymrm models - several domains or phases in one monolithic Newton system (membrane reactors, gas-liquid contactors, retentate/permeate, washcoat and channel), pressure-velocity coupling (Darcy, Ergun, velocity that varies with the state), nested scales (resolved particles in a reactor, Schur complement), and when a segregated solve is acceptable. Use when a pymrm model couples domains, phases or scales, or solves pressure and flow together.
---

# Coupling patterns in pymrm

Load the `conventions` skill first; its `assembly-styles.md` defines the block
assembly used here. Each pattern has a tested exemplar in `exemplars/` (paths
relative to this skill); copy its structure, then rebuild its checks for your
physics.

| Pattern | Exemplar | Key pymrm pieces |
|---|---|---|
| Several domains in one Newton system, each with its own operators and fields | `exemplars/membrane_reactor.py` (countercurrent retentate and permeate) | sub-shapes per domain, `update_array_indices` with rectangular coupling blocks, constant blocks placed once |
| Pressure and flow solved together; velocity depends on the state | `exemplars/darcy_gas_reactor.py` (Darcy flow, ideal gas, reaction) | face velocity from `construct_grad` of pressure; flux as a product, Jacobian by the product rule |
| Nested scales: a particle (or film, washcoat) at every reactor cell | `exemplars/nested_particle_bed.py` (plug-flow bed, resolved spheres, film) | `construct_grad(..., shapes_d=...)` for the bulk-dependent surface condition; Schur complement as a custom `newton` solver |

## Monolithic first

Solve coupled equations in ONE Newton system unless there is a reason not to.
A monolithic solve converges quadratically for the whole problem, needs no
outer iteration, and makes "is the coupling converged?" the same question as
"is the model converged?".

A segregated solve (domain by domain, iterating between them) is acceptable
when the coupling is one-way or weak (a trace species that does not change the
flow), when the time scales are so different that one domain is quasi-steady
for the other, or to reuse an existing solver for one part. Its costs: the
outer iteration may converge slowly or not at all for strong coupling, and the
reported solution is only as good as the outer tolerance. If you segregate,
report the coupling residual of the final state (evaluate the monolithic
residual on it with `residual_check`) and say why monolithic was not used.

## Building a monolithic model

1. **State layout.** Spatial axes first, fields last (style guide). Domains on
   the same grid can share one state with the domains as field slices (the
   membrane exemplar: `(n, 3)` = retentate A, retentate B, permeate B). Domains
   on different grids get their own sub-shapes placed in one flat vector (the
   nested exemplar: bulk `(n_z, 1)` then particles `(n_z, n_r, 1)`).
2. **Operators per domain,** on the domain's own sub-shape, grid and boundary
   conditions. Write each bc's physical equation beside it (pitfalls P1).
3. **Coupling blocks.** Interphase transfer, membrane fluxes and source terms
   that connect domains are blocks with rows of one domain and columns of the
   other; place them with `update_array_indices(block, (rows_shape,
   cols_shape), full_shape, offset=(row_offset, col_offset))`.
4. **Place constant blocks once.** State-dependent local blocks: `NumJac` on the
   stacked state, or a precomputed placed pattern (`assembly-styles.md`).
5. **Check the Jacobian** once with `check_jacobian` at a state away from the
   solution. A missing coupling block still converges, slowly or wrongly.
6. **Scale the unknowns** so every field is of order one (pressure in bar or
   relative to a reference, not Pa; pitfalls P7).

Two domains that meet at a shared face with continuity of value and flux (a
membrane resolved as its own domain, a washcoat on a channel wall): see
`construct_interface_matrices` in the installed pymrm (read its docstring); it
returns the interface coupling for both sides in the same way `construct_grad`
returns boundary terms.

## Pressure-velocity coupling

- The face velocity comes from the pressure gradient: Darcy `u = -K/mu dp/dz`
  is `-K/mu (grad @ p + grad_bc)` on the faces. Ergun adds a term quadratic in
  `u`; solve the face law for `u` or give `NumJac` the face relation.
- Convective fluxes become products of two state-dependent factors,
  `F = u(p) * c_face(x)`. With `c_face = C @ x + C_bc` (an upwind interpolation
  built once with `v = 1`), the Jacobian is `diag(u) @ C` for `x` and
  `diag(c_face) @ dU/dp` for `p`: exact, from constant matrices.
- Upwinding depends on the sign of `u`. If the flow can reverse, rebuild the
  interpolation per iteration or build both directions and select per face.
- Total continuity with an equation of state (ideal gas `c_t = p / (R T)`) closes
  the pressure. Freezing the velocity at its inlet value ("incompressible
  shortcut") is a break row, not a model: in the exemplar it moves the
  conversion by 6 %.

## Nested scales

- The inner problem's boundary condition depends on the outer unknown (film:
  `D dc/dr = k_f (c_b - c_s)`). Use `construct_grad(..., shapes_d=(None,
  shape_d))`: the returned `grad_bc_right` multiplies the outer vector. With
  `shapes_d` the dictionary's `d` is the COEFFICIENT on that vector (`d = Bi`,
  not `Bi * c_b`; pitfalls P5), and `shape_d` must broadcast to the boundary
  shape.
- The inner Jacobian is block diagonal (one particle per outer cell). Eliminate
  it with a Schur complement inside the linear solve: pass a function as
  `newton(..., solver=...)` that receives the blocks and returns the update.
  `splu` of the block-diagonal inner Jacobian is cheap. When each particle
  couples only to its own outer cell, the Schur correction is DIAGONAL and one
  solve against the summed coupling columns gives it (see the exemplar); never
  form it densely.
- On a 1-D outer grid a monolithic sparse LU of the full system is usually as
  fast or faster (0.04 s against 1.3 s in the exemplar). The Schur complement
  pays off for large inner problems, a multi-dimensional outer grid, or when the
  inner solver is reused elsewhere.
- Check the elimination against the full monolithic sparse solve of the same
  system once (a consistency check, not an independent route), and the physics
  against a limit with a closed form (linear kinetics in the exemplar).

## Checks that coupled models need

- **Balances across domains:** what leaves one domain enters the other. With
  face values and the same operators these are usually structural (exact by
  construction): report them as code checks, not as evidence.
- **An independent route in a limit:** plug flow as an ODE boundary-value
  problem (`solve_bvp`), a linear case with a closed form, one domain switched
  off.
- **Break rows that flip the coupling:** co-current instead of countercurrent,
  the film removed, the velocity frozen. If the checked numbers do not move, the
  checks do not see the coupling.
