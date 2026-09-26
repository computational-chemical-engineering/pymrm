# Assembling the Jacobian: three styles

Every pymrm model returns `(g, jac)`. How `jac` is built is a design choice with
consequences for correctness risk, flexibility, teaching value and, less than
one might think, speed. The exemplar `exemplars/assembly_styles.py` implements
one model (nonisothermal sphere pellet, fields `c` and `T`) in all styles; CI
requires the solutions to agree within 1e-10 and the Jacobians within 1e-5
(observed: 1e-13 and 3e-10).

## What the styles cost (measured on the exemplar)

Time per residual-and-Jacobian evaluation, and one sparse direct solve of the
same system, at n = 20 000 cells (40 000 unknowns):

| Variant | ms |
|---|---|
| A, operator sum (NumJac on the local source) | 2.4 |
| B, block assembly, blocks re-placed with `update_array_indices` every call | 6.4 |
| B, block assembly, placed pattern precomputed, values scattered per call | 1.1 |
| C, full-residual NumJac | 4.1 |
| one `spsolve` of the Jacobian | 18 |

Per Newton iteration the linear solve dominates, so all variants land within
about 25 % of each other here. Choose the style for correctness, clarity and
flexibility first. Assembly speed matters when the solve is cheap (small or
banded systems, factorisation reused across steps) or the evaluation count is
huge (sweeps, fits): then use A, or B with a precomputed pattern, and avoid
re-placing blocks in the inner loop.

## A, operator sum: the simplest, hard to get wrong

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

The default for a single domain whose fields share a grid. Every term is
visible and nameable, and the `NumJac` stencil is the default one.

## B, block assembly: multi-field and multi-domain models

Each field and each coupling is its own block, built on its own sub-shape with
its own operators and boundary conditions, and placed into the monolithic state
with `update_array_indices(block, (rows_shape, cols_shape), full_shape,
offset=(row_offset, col_offset))`. Off-diagonal (coupling) blocks are
rectangular. This is how models with different fields, grids or domains are
built: species, pressure and temperature with their own operators; retentate,
membrane and permeate; a particle coupled to a reactor. It is the natural style
for monolithic models; the `model-patterns` skill has tested exemplars.

Keep it fast:
- place every CONSTANT block once, at setup;
- for state-dependent blocks, either compute them with `NumJac` on the stacked
  state (A inside the block structure; as fast as A), or precompute the placed
  sparsity pattern once and scatter new values into it each call (the
  `BlockPattern` class in the exemplar; the fastest variant measured);
- avoid calling `update_array_indices` and summing sparse blocks on every
  Jacobian evaluation: that is the 6.4 ms row above.

`update_csc_array_indices` is deprecated; use `update_array_indices`.

## C, full-residual NumJac

`NumJac` differentiates the whole residual, transport included. The stencil must
declare every coupling: neighbours along each spatial axis (`axes_diagonals=[0]`
for axis 0, which covers offsets -1, 0, +1 only) and all fields within a cell
(`axes_blocks=[-1]`), on a shape with at least two axes (pitfalls P4).

Use it when a flux is nonlinear in the state and its analytic derivative is
awkward: Maxwell-Stefan and dusty-gas fluxes, concentration-dependent
diffusivity, electrochemical fluxes. Also acceptable for a quick prototype.

Not for TVD flux limiters: a limited face value reads two cells upwind, wider
than the stencil `axes_diagonals` can declare; a van Leer residual gave a 38 %
Jacobian error with no warning. Use deferred correction instead (`api-map.md`,
structure S5).

Costs and risks:
- An incomplete stencil gives a WRONG Jacobian without warning. In the exemplar,
  the default stencil on the full residual made the Jacobian singular and Newton
  failed; in milder cases Newton converges slowly or to a wrong answer.
- Evaluation is 1.4 to 1.8 times A in the measurements, growing with more fields
  and spatial dimensions.
- The Jacobian is only as accurate as the finite-difference step.

Hybrid, often best for nonlinear fluxes: keep the linear transport in the
constant matrix and give `NumJac` only the nonlinear flux part, with the
neighbour stencil.

## Choosing

| Situation | Style |
|---|---|
| One domain, fields on one grid, transport linear in the state | A |
| Fields with their own operators, grids or domains; monolithic coupling | B, with constant blocks placed once and local blocks by NumJac or a fixed pattern |
| Flux nonlinear in the state (Maxwell-Stefan, D(c), electrochemistry) | C, or the A + C hybrid |
| Velocity varying along the reactor (mole change, pressure drop, density) | A or B with the convection operator rebuilt or scaled per iteration, or treated as a coupled field; do not freeze a velocity that depends on the state |
| Small algebraic system (a CSTR, a point) | analytic Jacobian, or `NumJac((1, n_unknowns))` |
| Teaching the structure of the discretisation | A or B: every term is a named matrix |

Whatever the style, check the Jacobian once against a finite-difference Jacobian
of the full residual at a state away from the solution (the exemplar's check 2).
A wrong Jacobian can still converge, slowly or to the wrong root.
