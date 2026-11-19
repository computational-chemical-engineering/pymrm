# Which pymrm function for which job

The installed pymrm is the authority on signatures. Find and read it:

```bash
python -c "import pymrm, pathlib; print(pathlib.Path(pymrm.__file__).parent)"
python -c "import pymrm; help(pymrm.construct_grad)"
```

Every public function has a NumPy-style docstring. Read the docstring before
using a function you have not used in this session; do not rely on memory.

Gallery evidence is cited as "(gl 7, 41)": rows 7 and 41 of
`docs/refresh-learnings.md` in the pymrm-gallery repository; `A2.4` and the like
are its catalogue IDs, the pages that show a pattern working.

## Core recipe

Every model is: grid, boundary conditions, operators assembled once, a residual
that returns `(g, jac)`, and `newton` (or a direct sparse solve for a linear
model). See `style-guide.md` sections 7 to 11 and the exemplars.

| Job | Function | Notes |
|---|---|---|
| face and centre coordinates | `np.linspace` for faces, midpoints for centres; `non_uniform_grid(left, right, n, dx_inf, factor)` to cluster cells at a boundary | faces `x_f` have `n + 1` entries |
| diffusive gradient at faces | `construct_grad(shape, x_f, x_c, bc, axis)` | returns `(grad, grad_bc)`; with `shapes_d`, `(grad, grad_bc_left, grad_bc_right)` (pitfalls P5) |
| divergence of face fluxes | `construct_div(shape, x_f, nu, axis)` | `nu`: 0 slab, 1 cylinder, 2 sphere, or a callable area profile (structure S13) |
| upwind convective flux | `construct_convflux_upwind(shape, x_f, x_c, bc, v, axis)` | `v` at faces, scalar or broadcastable array. A state-dependent velocity passed cell-centred is interpolated and extrapolated to the inlet face, so the inlet admits `v_f[0] c_in`, not the feed (`F2.3`: 0.99229 of the feed, conversion 93.10 % against 92.62 % from the reaction integral): convect the molar flux `q = s(state) y` with a constant unit-velocity operator instead (gl 184) |
| higher-order (TVD) convection, steady or Newton | `ConvFluxTVD(shape, x_f, x_c, bc, tvd_limiter, axis)`, called as `flux, jac = tvd(c, v)` | face fluxes and their exact Jacobian; `shapes_d` and `jac_v=True` add the Jacobians to the boundary data and the velocity; see below (pitfalls P10) |
| TVD face values for deferred correction in a time step | `interp_cntr_to_stagg_tvd(c, x_f, x_c, bc, v, tvd_limiter, axis)` | returns `(face_values, correction)`; see below |
| per-cell or per-face coefficients as a matrix | `construct_coefficient_matrix(coef, shape, axis)` | face diffusivities, accumulation weights, sinks on selected cells (pitfalls P6); the exact diagonal Jacobian of a pointwise rate (pitfalls P9); per-cell field blocks, see "Domains in one state" |
| values and gradients at the boundaries | `compute_boundary_values(c, x_f, x_c, bc, axis)` | gradients along `+axis`, not the outward normal (pitfalls P1, P8). Accepts extra axes: the boundary flux of every step of a march from one call on the stacked profiles `(n_x, n_t)` (`A3.2`: one call in place of one per step at about 0.1 ms each) (gl 85) |
| face values from centres | `interp_cntr_to_stagg`, `interp_stagg_to_cntr` | linear interpolation |
| Jacobian of a nonlinear source | `NumJac(shape)`, called as `g, jac = numjac(f, c)` | shape keeps a field axis (pitfalls P3, P4) |
| nonlinear solve | `newton(fun, x0, tol, maxfev, solver, callback)` | `fun(x) -> (g, jac)`; absolute step tolerance (pitfalls P7); direct LU at every size; `result.status`, `result.backward_error` |
| hard nonlinear solve | `newton(..., damping="guarded"/"monotone", scale=..., bounds=..., backward_tol=1e-8, stagnation=True, refresh=...)` (from 3.0.0) | see "Hard nonlinear problems" below |
| keep iterates where the residual is defined | `newton(..., bounds=(lower, upper))` (from 3.0.0); older: `clip_approach(values, dummy, lower_bounds=0)` as the `newton` callback | bounds for `T > 0`, `log`, `sqrt`; prefer kinetics smooth through zero over a bound at zero |
| steady state by pseudo-time stepping | `pseudo_transient(fun, x0, mass, dt)` (from 3.0.0) | when Newton from the guess fails; `mass` in the units of the time term. Needs no parameter path, so it suits a sweep over many cases (`B1.6`); floor unknowns that overflow on the way (`theta >= 0.2` for `exp(gamma/theta)`, inactive at every solution) (gl 112) |
| solutions along a parameter | `continuation(fun_p, x0, params)` (from 3.0.0) | warm starts with a secant predictor; stops at a fold instead of jumping branches. Around folds: see "Branches, folds and thresholds" below. Up to 3.1.0 two nearly equal values in `params` (`0.1` and `1 - 0.9`) can stop the march with a false fold report: drop such repeats there; later versions retry with a bounded predictor. Up to 3.3.0 the halvings of all sub-steps towards one value share one `max_halvings` budget, so a long stretch of short sub-steps stops with a false fold report (request closer values there), and a residual that raises at the predicted state ends the march (return a NaN residual instead); later versions count halvings per sub-step and halve on a raise |
| see what a bc dictionary means | `describe_bc(bc, x_f)` (after 2.3.1) | prints each end as an equation in x (pitfalls P1) |
| stirred-volume or tank outlet | `{"outflow": True}` as the bc dictionary (after 2.3.1) | face value = adjacent cell (pitfalls P6). Never leave an outlet `None`: that face carries no flux (a Dirichlet zero), and with an exchange or reaction sink newton converges silently to a wrong state; later than 3.3.0 it warns |
| Jacobian, refinement, threshold and residual checks | `pymrm.checks` (after 2.3.1; copy in `scripts/pymrm_checks.py`) | see `numerics.md`. Root-find on the physical range with `find_roots` and demand the expected number of roots, not `brentq` on a hand bracket: `B2.1` printed a conversion of 210 % for a mistyped constant that `find_roots` on 0 to 100 % refuses (also `J4.4`). A quantity with a pole in the range gives a spurious sign change; multiply by the denominator first (`F1.3`) (gl 173, 207, 301) |
| couple sub-domains in one system | `BlockLayout(domains, axis=...)`: name the domains once, place blocks by name (`place`, summed with `+`), reach a domain's part of a vector (`view`, `put`, `pick`); `construct_interface_matrices` | structure S7; see "Domains in one state" below and `assembly-styles.md`; `update_array_indices` is the low-level tool underneath; `update_csc_array_indices` is deprecated |
| couple the fields of one cell (exchange, equilibrium, reaction between fields) | `construct_coefficient_matrix(M, shape=((n, k, 1), (n, 1, k)))` with `M` a `(k, k)` (or per-cell `(n, k, k)`) matrix | one small matrix per cell on an `(n, k)` state; see "Domains in one state" below |
| boundary value or boundary rows as an operator (film, wall reaction, bulk source) | `construct_boundary_value_matrices`; `pick` of a `BlockLayout` over the face space (or an identity placed with `update_array_indices`) selects boundary-face rows | never compute flat indices by hand; see "Reaching a boundary" below |

## Domains in one state: `BlockLayout`

A monolithic model stacks domains (particle and bulk, wall and fluid, channels,
fields on one grid) into one state. `BlockLayout` names them once; the state is
`np.concatenate(parts, axis)` of the domain arrays (`axis=None`: region-major,
the concatenation of the flattened domains, for domains with different field
counts). Offsets follow from the concatenation; no flat index is typed.

```python
layout = BlockLayout({"particle": (n_z, n_r, m), "bulk": (n_z, 1, m)}, axis=1)
lin = (layout.place(div_r @ flux, "particle", "particle")      # blocks on the
       + layout.place(div_r @ flux_b, "particle", "bulk")      # domains' own shapes,
       + layout.place(bulk_op, "bulk", "bulk")
       + layout.place(to_bulk, "bulk", "particle")).tocsc()
g_0 = np.zeros(layout.size); layout.put(g_0, "bulk", (div_z @ conv_bc).toarray())
c_b = layout.view(c, "bulk")                       # a view, never a copy
```

- `place(block, rows, cols)` checks the block size against the named domains
  (`update_array_indices` itself silently accepts a wrong size); `None` for a
  side leaves it as is. Sum placed blocks with `+`; for a system of many blocks
  (dozens or more) `assemble([(block, rows, cols), ...])` builds the sum in one
  pass (3x faster at 144 blocks; slower than `+` for a handful).
- Fields as domains: `BlockLayout.from_fields(["psi", "xi"], (n,))` for an
  `(n, 2)` state (short for `{"psi": (n, 1), "xi": (n, 1)}, axis=-1`; the
  trailing 1 is what interleaves: `(n,)` domains with `axis=-1` would stack to a
  `(2n,)` state); a one-field operator placed at `("psi", "psi")`.
- Part of a domain: `layout.subdomain("near_wall", "fluid", np.s_[:, 0])` names
  a box of a domain (the first fluid row) without adding unknowns; use the name
  in `place`, `view`, `put`, `add`, `pick` like a domain. For example a wall
  rate evaluated in the first fluid cell: `place(-D * (div_y @ grad_r),
  "fluid", "near_wall")`.
- Residual parts: `put(g, name, values)` writes, `add(g, name, values)`
  accumulates (a term that acts on one domain only, e.g. TVD fluxes in the
  fluid rows). `assemble` drops explicit zeros, as `+` of sparse arrays does.
- `NumJac`: with an integer axis, `NumJac(layout.shape)` on the full state as
  usual. With `axis=None`, `NumJac` per domain on `layout.shapes[name]` applied
  to `layout.view(u, name)`, placed with `place(jac, name, name)`; never
  `NumJac` on the flat `(size,)` shape (dense Jacobian, pitfalls P3).
- Coupling between the fields of each cell (interphase exchange
  `kappa (x_0 - x_1)`, an equilibrium, a reaction A -> B) is one small matrix
  per cell: `construct_coefficient_matrix(kappa * np.array([[1, -1], [-1, 1]]),
  shape=((n, 2, 1), (n, 1, 2)))` gives the `(2n, 2n)` block with rows and
  columns `2i, 2i + 1` for an `(n, 2)` state; `kappa` may be per cell, shape
  `(n, 1, 1)`. For a rectangular coupling between a `k`-field and an
  `l`-field domain use `((n, k, 1), (n, 1, l))` and a `(k, l)` matrix.
- The same call with `dS[i, a, b] = dS_a/dc_b` is the exact Jacobian of a
  pointwise multi-field source, and `construct_coefficient_matrix([D, D2],
  shape=(n, 2), axis=0)` gives the face diffusivities of two fields. Pages to
  copy: `A2.5` (exchange, identical to the hand-built `coo_array`), `A3.3`
  (2 to 3 iterations per step, `check_jacobian` 1.9e-8), `B1.6` (90 against
  150 pseudo-time steps with `NumJac((n, 2))`), `B3.2` (a complementarity row
  `min(step residual, xi) = 0` with `from_fields`). A state gap between an
  exact-Jacobian and a `NumJac` run cannot test the Jacobian, since newton
  converges to the root whatever Jacobian it gets: compare the two Jacobians
  directly with `check_jacobian` (gl 68, 90, 113, 136).

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
- What raises and what returns: non-convergence returns `success=False`; a
  singular Jacobian or a non-finite residual raises `NewtonBreakdown`, which a
  wrapper checking only `success` never sees. A residual that is flat to the
  last bit at the guess (`1 - (1 - g)**3 exp(...)` near 1) differences to an
  exactly zero `NumJac` column: `B3.3` raised at 13 of 20 states, the log form
  of the same residual converged at all 20 (gl 139).
- Passing a callable `solver=` (a step limiter, say) turns `damping="auto"`
  into `"none"`, as does a `callback`. Remove a custom step cap and test
  `bounds` and `damping` on their own before keeping it (`I1.2`) (gl 47).
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
For a lower round-off floor write the steady residual as the correction to the
upwind solution (pitfalls P10, `A2.1`) (gl 59). The term `jac_up @ (c - c_up)`
is the upwind residual only when the upwind problem is linear and `c_up` is its
solution, and its 1e-15 agreement with deferred correction is between two
routes sharing one LU factorisation, so it does not test the factorisation.

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
  them with `pick` of a `BlockLayout` over the faces, `outer =
  BlockLayout({"inner": (n_z, n_r, n_c), "outer": (n_z, 1, n_c)},
  axis=1).pick("outer")`, or with an identity placed by
  `update_array_indices`. For a field
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
value at `j = n_r` after the particle cells), declare the layout with
`BlockLayout` and place every block by domain name. A wall reaction `r(c_w)`: give the fluid the flux
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

**Branches, folds and thresholds.** Damping does not choose between multiple
solutions. An S-curve through both folds needs no arclength: pin a state value
(the pellet centre concentration) with one extra equation and make the physical
parameter (`phi**2`) an unknown, then `continuation` in the pinned value
(`B1.1`: 319 points through both folds in one call, 1e-5 from a shooting
reference). Locate a threshold by following the branch: one cold solve at a safe
value, then warm-start every scan point and bisection midpoint from the
converged state at the current lower end. Cold starts from a simpler model's
profile let round-off decide the branch (`D1.1`: 990.9 C, 411.5 C, failure,
992.1 C at four neighbouring parameter values) (gl 8, 16).

**A state-dependent face diffusivity is one Newton solve**, not a Picard loop:
`c_f` from `interp_cntr_to_stagg` between cells and `compute_boundary_values`
at the ends, the residual differenced with `NumJac((n, 1),
axes_diagonals=[0])` because a cell reads its neighbours through `c_f`
(`B1.3`: 5 iterations against 13 outer Picard iterations, same solution to
1.5e-14) (gl 103).

**Boundary data that depend on the state** (a flux condition with `t0+(c)` in
it): `construct_grad(..., shapes_d=((1, 1), None))` with a unit Neumann
dictionary, the datum computed in the residual and supplied as
`grad_bc_left @ [g]`; the operator is assembled once, and
`construct_boundary_value_matrices` with the same datum gives the face values.
The datum must use the same property values as the flux (`J3.4`: salt drift
1e-15 over 162 steps) (gl 200). Where the wall flux enters linearly, the
response to a unit wall flux is state independent and is computed once per step
length, which makes each implicit step one linear solve and a scalar root
(`I1.3`) (gl 51).

**A nonlinear interface condition** (an equilibrium `p2 = f(c2)`) needs the
interface values as unknowns: `(1, 1)` domains of a `BlockLayout`, `shapes_d`
boundary values, and the interface face of each film a `subdomain` of a face
layout selected with `pick`. `construct_interface_matrices` covers linear
conditions only. Give the nonlinear row its exact derivative (`A3.1`: Newton
fails within 50 iterations without it, 3 to 6 with it) (gl 80).

**Streams in opposite directions** (bubble and cloud up, emulsion down) cannot
share one `construct_convflux_upwind` call: `{"outflow": True}` marks a whole
boundary. Build one `(n, 1)` operator per stream, the inlet values as extra
unknowns through `shapes_d` and the end mixing as extra rows: one sparse solve,
no fixed-point loop over the end values (`E2.1`: balance 1.7e-12) (gl 165).

**Plug flow coupled to lumped unknowns** (a well-mixed liquid): one `newton`
system of the profile plus the lumped values, the height integral as dense
coupling rows of the Jacobian, in place of a relaxed loop inside `fsolve`. Build
per-cell blocks with `bsr_array((blocks, np.arange(n), np.arange(n + 1)))`:
`block_diag` took 83 % of `F2.3`'s run time (0.85 s against 0.15 s) (gl 186).

**A march whose step matrix changes only through the step length**: store the
constant tridiagonal operator once in `solve_banded` layout and add `cap/dz` to
its diagonal each step (`A3.2`: 0.036 s against 0.147 s with per-step sparse
assembly and 0.31 s with `spsolve`) (gl 85).

**A pure-Neumann problem pinned at one cell** is consistent only if its source
has zero mean on the grid: subtract `sum(w * u)` with the cell volumes the
divergence conserves, not the exact mean, and test the pin at two cells
(`A2.3`: pin axis against wall 1.5e-4 with the exact mean, 1.1e-13 with the grid
mean) (gl 62).

## More worked examples

The pymrm gallery reproduces published reactor-engineering results with pymrm,
each page with its checks:
<https://github.com/computational-chemical-engineering/pymrm-gallery> (see
`pages/`, and `docs/taxonomy.md` for the structure codes). Treat gallery code as
an example of physics and checks; the conventions here take precedence where
they differ.
