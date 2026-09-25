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
| higher-order convection | `interp_cntr_to_stagg_tvd(c, x_f, x_c, bc, v, tvd_limiter, axis)` | returns `(face_values, correction)`; deferred correction, see below |
| per-cell or per-face coefficients as a matrix | `construct_coefficient_matrix(coef, shape, axis)` | face diffusivities, accumulation weights, sinks on selected cells (pitfalls P6) |
| values and gradients at the boundaries | `compute_boundary_values(c, x_f, x_c, bc, axis)` | gradients along `+axis`, not the outward normal (pitfalls P1, P8) |
| face values from centres | `interp_cntr_to_stagg`, `interp_stagg_to_cntr` | linear interpolation |
| Jacobian of a nonlinear source | `NumJac(shape)`, called as `g, jac = numjac(f, c)` | shape keeps a field axis (pitfalls P3, P4) |
| nonlinear solve | `newton(fun, x0, tol, maxfev, solver, callback)` | `fun(x) -> (g, jac)`; absolute step tolerance (pitfalls P7) |
| keep iterates physical | `clip_approach(values, dummy, lower_bounds=0)` as the `newton` callback | |
| couple sub-domains in one system | `update_array_indices` (place a block into a larger state), `construct_interface_matrices` | structure S7; `update_csc_array_indices` is deprecated; see `assembly-styles.md` |

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

## Patterns beyond the core recipe

**Deferred correction for TVD convection** (structure S5). Keep first-order
upwind in the constant Jacobian; add the limiter correction to the residual:

```python
conv, conv_bc = construct_convflux_upwind(shape, x_f, x_c, bc, v, axis=0)
jac = accum + div @ conv                          # factorise once
c_f, dc_f = interp_cntr_to_stagg_tvd(c, x_f, x_c, bc, v, tvd_limiter=minmod, axis=0)
g = jac @ c.reshape(-1, 1) - c_old.reshape(-1, 1) / dt + div @ (conv_bc + v * dc_f.reshape(-1, 1))
```

One or two inner iterations per time step. Validation: sharper than first-order
upwind and still bounded.

**Apparent rate of a particle from its surface flux.** Take the surface-face
rows of the flux operator rather than re-integrating the source; with
`A/V = (nu + 1)/R` the effectiveness factor follows. The volume integral of the
source equals this flux identically (the divergence telescopes), so comparing
the two is a structural identity, not a check.

**Extra unknowns at a boundary** (surface reaction, film). Pass
`shapes_d=((1, n_c), None)` to `construct_grad`; the rectangular
`grad_bc_left` couples the interior to a boundary unknown whose own equation is
the surface balance.

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
