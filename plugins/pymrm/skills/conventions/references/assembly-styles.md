# Assembling the Jacobian: three styles

Every pymrm model returns `(g, jac)`. How `jac` is built is a design choice with
consequences for speed, flexibility, correctness risk and teaching value. The
exemplar `exemplars/assembly_styles.py` implements one model (nonisothermal
sphere pellet, fields `c` and `T`) in all three styles; CI checks that their
solutions agree to 1e-13 and their Jacobians to 3e-10.

## The default: A, operator sum

The transport Jacobian comes from the operators (`construct_grad`,
`construct_div`, `construct_convflux_upwind`), assembled ONCE. `NumJac`
differentiates only local terms: reactions, interphase mass and heat transfer,
pointwise equilibria. The Jacobian is a sum:

```python
# setup
jac_const = accumulation + div @ (conv - diff * grad)        # constant
numjac = NumJac(shape)                                       # (n, n_c): block per cell
# every Newton iteration
g_loc, jac_loc = numjac(source, u)
g = g_const + jac_const @ u.reshape(-1, 1) - g_loc.reshape(-1, 1)
jac = jac_const - jac_loc
```

Use it by default. It is the fastest of the three in every measurement so far,
each term of the Jacobian is visible and nameable, a linear model can factorise
`jac_const` once, and the `NumJac` stencil is the default one, which is hard to
get wrong.

## B, block assembly (multi-field and multi-domain models)

Each field and each coupling is its own block, built on its own sub-shape with
its own operators and boundary conditions, and placed into the monolithic state
with `update_array_indices(block, (rows_shape, cols_shape), full_shape,
offset=(row_offset, col_offset))`. Off-diagonal (coupling) blocks are
rectangular. This is how models with different fields on different grids or
domains are built: species, pressure and temperature with their own operators;
retentate, membrane and permeate; a particle coupled to a reactor. See the
`model-patterns` skill for the monolithic coupling patterns.

Rules that keep it fast:
- Place every CONSTANT block once, at setup, into a constant monolithic matrix.
- Re-placing state-dependent blocks on every call is expensive: in the exemplar,
  re-placing four diagonal blocks per call made B 2.8 to 6.6 times slower than A
  (from 1.15 ms against 0.18 ms at n = 200 to 6.5 ms against 2.4 ms at
  n = 20 000). Where the state-dependent
  part is local to a cell, compute it with `NumJac` on the stacked state (style A
  inside the block structure), or precompute the placed sparsity pattern and only
  update its values.
- `update_csc_array_indices` is deprecated; use `update_array_indices`.

## C, full-residual NumJac

`NumJac` differentiates the whole residual, transport included. The stencil must
declare every coupling: neighbours along each spatial axis
(`axes_diagonals=[0]` for axis 0) and all fields within a cell
(`axes_blocks=[-1]`), on a shape with at least two axes (pitfalls P4).

Use it when a FLUX is nonlinear in the state and its analytic derivative is
awkward: Maxwell-Stefan and dusty-gas fluxes, concentration-dependent
diffusivity, electrochemical fluxes, flux limiters inside Newton. Also
acceptable for a quick prototype.

Costs and risks:
- Slower: 1.4 to 1.8 times A in the measurements (3.9 ms against 2.4 ms at
  n = 20 000), because `NumJac` must perturb more column groups when the stencil
  covers neighbours. The gap grows with more fields and more spatial dimensions.
- An incomplete stencil gives a WRONG Jacobian without warning. In the exemplar,
  the default stencil on the full residual made the Jacobian singular and Newton
  failed; in milder cases Newton only converges slowly or to a wrong answer.
- The Jacobian is only as accurate as the finite-difference step.

Hybrid, often best for nonlinear fluxes: keep the linear transport in the
constant matrix (A) and give `NumJac` only the nonlinear flux part, with the
neighbour stencil.

## Choosing

| Situation | Style |
|---|---|
| One domain, transport linear in the state | A |
| Several fields with their own operators, grids or domains | B for the structure, A for local terms inside it |
| Flux nonlinear in the state (Maxwell-Stefan, D(c), electrochemistry) | C, or the A + C hybrid |
| Small algebraic system (a CSTR, a point) | analytic Jacobian, or `NumJac((1, n_unknowns))` |
| Teaching the structure of the discretisation | A or B: every term is a named matrix |

Whatever the style, check the Jacobian once against a finite-difference Jacobian
of the full residual at a state away from the solution (the exemplar's check 2).
A wrong Jacobian can still converge, slowly or to the wrong root.
