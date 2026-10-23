# Which pymrm function for which job

The installed pymrm is the authority on signatures. Find and read it:

```bash
python -c "import pymrm, pathlib; print(pathlib.Path(pymrm.__file__).parent)"
python -c "import pymrm; help(pymrm.construct_grad)"
```

Every public function has a NumPy-style docstring. Read the docstring before
using a function you have not used in this session; do not rely on memory.

## Core recipe

Every model is: grid, boundary conditions, operators assembled once, a residual
that returns `(g, jac)`, and `newton` (or a direct sparse solve for a linear
model). See `style-guide.md` sections 7 to 11 and the exemplars.

| Job | Function | Notes |
|---|---|---|
| face and centre coordinates | `np.linspace` for faces, midpoints for centres; `non_uniform_grid(left, right, n, dx_inf, factor)` to cluster cells at a boundary | faces `x_f` have `n + 1` entries |
| diffusive gradient at faces | `construct_grad(shape, x_f, x_c, bc, axis)` | returns `(grad, grad_bc)`; with `shapes_d`, `(grad, grad_bc_left, grad_bc_right)` (pitfalls P5) |
| divergence of face fluxes | `construct_div(shape, x_f, nu, axis)` | `nu`: 0 slab, 1 cylinder, 2 sphere, or a callable area profile (structure S13) |
| upwind convective flux | `construct_convflux_upwind(shape, x_f, x_c, bc, v, axis)` | `v` at faces, scalar or broadcastable array |
| higher-order (TVD) convection, steady or Newton | `ConvFluxTVD(shape, x_f, x_c, bc, tvd_limiter, axis)`, called as `flux, jac = tvd(c, v)` | face fluxes and their exact Jacobian; `shapes_d` and `jac_v=True` add the Jacobians to the boundary data and the velocity; see below (pitfalls P10) |
| TVD face values for deferred correction in a time step | `interp_cntr_to_stagg_tvd(c, x_f, x_c, bc, v, tvd_limiter, axis)` | returns `(face_values, correction)`; see below |
| per-cell or per-face coefficients as a matrix | `construct_coefficient_matrix(coef, shape, axis)` | face diffusivities, accumulation weights, sinks on selected cells (pitfalls P6) |
| values and gradients at the boundaries | `compute_boundary_values(c, x_f, x_c, bc, axis)` | gradients along `+axis`, not the outward normal (pitfalls P1, P8) |
| face values from centres | `interp_cntr_to_stagg`, `interp_stagg_to_cntr` | linear interpolation |
| Jacobian of a nonlinear source | `NumJac(shape)`, called as `g, jac = numjac(f, c)` | shape keeps a field axis (pitfalls P3, P4) |
| nonlinear solve | `newton(fun, x0, tol, maxfev, solver, callback)` | `fun(x) -> (g, jac)`; absolute step tolerance (pitfalls P7); direct LU at every size; `result.status`, `result.backward_error` |
| hard nonlinear solve | `newton(..., damping="guarded"/"monotone", scale=..., bounds=..., backward_tol=1e-8, stagnation=True, refresh=...)` (from 3.0.0) | see "Hard nonlinear problems" below |
| keep iterates where the residual is defined | `newton(..., bounds=(lower, upper))` (from 3.0.0); older: `clip_approach(values, dummy, lower_bounds=0)` as the `newton` callback | bounds for `T > 0`, `log`, `sqrt`; prefer kinetics smooth through zero over a bound at zero |
| steady state by pseudo-time stepping | `pseudo_transient(fun, x0, mass, dt)` (from 3.0.0) | when Newton from the guess fails; `mass` in the units of the time term |
| solutions along a parameter | `continuation(fun_p, x0, params)` (from 3.0.0) | warm starts with a secant predictor; stops at a fold instead of jumping branches |
| see what a bc dictionary means | `describe_bc(bc, x_f)` (after 2.3.1) | prints each end as an equation in x (pitfalls P1) |
| stirred-volume or tank outlet | `{"outflow": True}` as the bc dictionary (after 2.3.1) | face value = adjacent cell (pitfalls P6) |
| Jacobian, refinement, threshold and residual checks | `pymrm.checks` (after 2.3.1; copy in `scripts/pymrm_checks.py`) | see `numerics.md` |
| couple sub-domains in one system | `update_array_indices` (place a block into a larger state), `construct_interface_matrices` | structure S7; `update_csc_array_indices` is deprecated; see `assembly-styles.md` |
| boundary value or boundary rows as an operator (film, wall reaction, bulk source) | `construct_boundary_value_matrices`; an identity placed with `update_array_indices` selects boundary-face rows | never compute flat indices by hand; see "Reaching a boundary" below |

## `NumJac` stencils

- `NumJac(shape)`: full coupling within the LAST axis, none in space. Right for a
  pointwise `reaction(c)` with `c` of shape `(n, n_c)`.
- `NumJac(shape, axes_blocks=[-2, -1])`: couples phases and species at a point
  (interphase transfer inside the source term).
- `NumJac(shape, axes_diagonals=[0])`: adds neighbour coupling along axis 0 when
  the SOURCE TERM reads neighbouring cells. Only for shapes with at least two
  axes (pitfalls P4).
- `NumJac(shape_in=..., shape_out=...)`: rectangular blocks, for example the
  derivative of species residuals with respect to temperature.

## `newton` options

- `solver="splu"`: `fun` must then return an already factorised `SuperLU`
  object. Use it when the Jacobian is constant and can be factorised once.
- `maxfev=10` per time step is normal with the previous step as the guess.
- Always check `result.success` AND the final residual norm.
- The linear solve is a direct sparse LU at every size (from 3.0.0; before,
  `solver=None` switched to ILU-BiCGSTAB above 50 000 unknowns). Iterative
  solvers only on request: `solver="bicgstab"`. Keep LU in 1-D and 2-D; for a
  3-D model pass an object with `factorize(jac)` returning an object with
  `.solve(rhs)`, for example AMG-preconditioned CG (example
  `newton_linear_solvers.ipynb`: at 50^3 cells AMG 2.1 s, ILU-GMRES 48 s,
  LU 432 s; at 200^2 LU 0.65 s, AMG 0.58 s).
- `result.fun` is the residual of the last evaluated iterate: for a converged
  solve the one before the final step (as before). `result.status` says why it
  stopped; `result.history` holds the step norm and damping per iteration
  (from 3.0.0).
- A singular Jacobian or a non-finite residual or step raises (as before);
  from 3.0.0 as `pymrm.NewtonBreakdown`, a `ValueError` whose `.result` holds
  the status and the last iterate. Non-convergence is `result.success=False`.

### Hard nonlinear problems (from 3.0.0)

Try these in order; each is opt-in and documented in the `newton` docstring.

1. Non-finite residual after a full step (overflow in an Arrhenius term,
   `log` of a negative): `damping="guarded"`. Same steps as plain Newton
   wherever those work.
2. Full steps overshoot and oscillate (dead zones, half-order kinetics):
   make the kinetics smooth and first order through zero
   (`k c (c**2 + eps**2)**((n - 1)/2)` instead of `k c**n`), then
   `damping="monotone"`. A hard bound at zero stalls instead (status 8).
3. Unknowns of different units in one state: `scale=` the typical magnitude
   per field; tolerances and damping then weigh them equally.
4. Trace unknowns and the absolute `tol` (pitfalls P7): `backward_tol=1e-8`.
5. `tol` below the round-off floor (switching terms, upwinding):
   `stagnation=True` with `backward_tol`.
6. Quantities lagged outside the Jacobian: `refresh=` instead of `callback=`.
7. Still failing from a poor guess: `pseudo_transient(fun, x0, mass, dt)`.
   Several steady states: `continuation(fun_p, x0, params)` from a known
   solution; it reports a fold rather than jumping branches.

## Patterns beyond the core recipe

**TVD convection with Newton.** `ConvFluxTVD` returns the TVD face fluxes and
their exact Jacobian: the upwind part, the limiter derivative and the boundary
faces. Build it once, call it in the residual:

```python
tvd = ConvFluxTVD(shape, x_f, x_c, bc=bc, tvd_limiter=vanleer, axis=0)

def model(c):
    flux, jac = tvd(c, v)
    return div @ flux.ravel() + other @ c + g_0, div @ jac + other

sol = newton(model, c0, damping="monotone")      # undamped Newton can cycle (P10)
```

Every limiter converges in about 5 to 25 steps. With `shapes_d` the call takes
`d=(ext_left, ext_right)` and also returns `jac_bc_left, jac_bc_right`
(derivatives only: `flux` already holds the boundary data). With `jac_v=True` it
appends `jac_v`, whose entries are the face values, for a velocity that depends
on the state: for Darcy flow `v = -(K/mu) (grad @ p + grad_bc)` the block is
`div @ jac_v @ (-(K/mu) * grad)`. `v` is a face velocity, accepted in the same
forms as by `construct_convflux_upwind`; the columns of `jac_v` follow the
shape of `v` as passed, and `jac_v=True` refuses a cell-centred `v`. An
initial guess with `v = 0` everywhere gives a zero convective Jacobian.

**Deferred correction for TVD convection** (structure S5), inside a time step:
keep first-order upwind in the constant Jacobian and add the limiter correction
to the residual:

```python
conv, conv_bc = construct_convflux_upwind(shape, x_f, x_c, bc, v, axis=0)
jac = accum + div @ conv                          # factorise once
c_f, dc_f = interp_cntr_to_stagg_tvd(c, x_f, x_c, bc, v, tvd_limiter=minmod, axis=0)
g = jac @ c.reshape(-1, 1) - c_old.reshape(-1, 1) / dt + div @ (conv_bc + v * dc_f.reshape(-1, 1))
```

One or two inner iterations per time step, at a small Courant number. Not for a
steady state: it converges slowly with minmod and stalls with van Leer, MUSCL
and SMART (pitfalls P10). Validation: sharper than first-order upwind and still
bounded.

**Apparent rate of a particle from its surface flux.** Take the surface flux
from the operators rather than re-integrating the source; with
`A/V = (nu + 1)/R` the effectiveness factor follows. The volume integral of the
source equals this flux identically (the divergence telescopes), so comparing
the two is a structural identity, not a check. For a 1-D particle the surface
is the last face along r (the last `n_c` rows of the flux operator for `n_c`
fields); for a field with more axes, reach the boundary as in the next
paragraph.

**Reaching a boundary without hand-computed indices.** Never build a selection
matrix or a flat offset by hand (`np.arange(n) * (n_r + 1) + n_r`,
`ravel_multi_index`): pymrm gives both pieces.
- The boundary ROWS of any face operator (a face gradient, a flux): select
  them with an identity placed by `update_array_indices`. For a field
  `(n_z, n_r, n_c)` and its outer face along axis 1:
  `outer = update_array_indices(eye_array(n_z * n_c), (shape_bnd, shape_bnd),
  (shape_bnd, shape_faces), offset=(None, (0, n_r, 0)))` with `shape_bnd =
  (n_z, 1, n_c)` and `shape_faces = (n_z, n_r + 1, n_c)`; the offset is the
  face index along the boundary axis (0 for the lower face, the number of cells
  for the upper one). Then `outer @ grad` and `outer @ grad_bc` are the surface
  gradient rows. This is the robust route for a film flux, also when a large Bi
  switches the film off.
- The boundary VALUE as an operator: `construct_boundary_value_matrices(shape,
  x_f, bc=bc_right, axis=..., bound_id=1, shape_d=...)` returns `(mat, mat_bc)`
  with rows on the boundary shape (`shape` with that axis set to 1), so
  `c_s = mat @ c + mat_bc @ c_ext`. With `shape_d` the condition reads
  `a dc/dn + b c_s = d c_ext`, so the flux could be written as
  `a dc/dn = d c_ext - b c_s`; for large `b/a` this cancels (digits are lost
  from about Bi = 1e6), so take the flux from the gradient rows above and use
  the value where a value is needed (a wall concentration in a rate).

**Extra unknowns at a boundary** (surface reaction, film, a wall
concentration). Keep them in the state as one more position along the
boundary-normal axis (a wall value at `j = 0` before the fluid cells, a bulk
value at `j = n_r` after the particle cells), and place every block with
`update_array_indices`. A wall reaction `r(c_w)`: give the fluid the flux
condition with `shapes_d` (`d` is the coefficient on the external field `r`),
and make the wall row `c_w - (mat @ c + mat_bc @ r(c_w)) = 0` with
`construct_boundary_value_matrices` for the same condition; this is second
order, while evaluating `r` at the wall-adjacent cell value is first order.

**Nested scales** (structure S8, reactor with particles). Solve the particle,
eliminate its interior with a Schur complement (`splu` of the particle Jacobian,
solve against the boundary coupling), so the reactor sees an apparent rate and
its derivative with respect to the bulk value.

**2-D models** (structure S6). Build `construct_grad` and `construct_div` once
per axis on the full `(n_z, n_r, n_c)` shape with the right `axis=` and add the
Jacobians. Use `nu=1` on the radial axis of a tube. Do not assemble Kronecker
products by hand.

**When plain Newton fails** (thin reaction layers, strongly exothermic or stiff
kinetics): pseudo-transient continuation (implicit steps with a growing `dt`)
followed by one exact steady solve, or parameter continuation. A number you
report must come from a deterministic solve, not from wherever a warm-start
chain happened to land; with multiplicity, locate every branch explicitly.

## More worked examples

The pymrm gallery reproduces published reactor-engineering results with pymrm,
each page with its checks:
<https://github.com/computational-chemical-engineering/pymrm-gallery> (see
`pages/`, and `docs/taxonomy.md` for the structure codes). Treat gallery code as
an example of physics and checks; the conventions here take precedence where
they differ.
