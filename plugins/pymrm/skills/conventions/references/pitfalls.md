# pymrm pitfalls

API behaviour that has produced wrong answers in real models. Every entry is
demonstrated by a test in `test/test_pitfalls.py` of the pymrm repository, which
pins the numbers quoted here. If one of those tests starts failing, the API has
changed and this file must change with it.

Several pitfalls were removed or guarded in the pymrm release after 2.3.1. Each
entry says what holds for that release ("new") and for 2.3.1 and older ("old").
Check the installed version: `python -c "import pymrm; print(pymrm.__version__)"`.

Each entry: what goes wrong, how you notice, what to do.

## P1. Boundary conditions use the OUTWARD normal

`bc = ({"a", "b", "d"}, {"a", "b", "d"})` means `a * dc/dn + b * c = d` with `n`
pointing out of the domain. At the right end `dc/dn = dc/dx`; at the left end
`dc/dn = -dc/dx`. The same dictionary gives opposite fluxes at the two ends.

- Symptom: a flux boundary condition drives material the wrong way; a profile
  comes out mirrored; a Danckwerts inlet with the wrong sign of `a` loses mass.
- Measured: `{"a": 1, "b": 0, "d": q}` at the LEFT end of a slab gives
  `c = q (1 - x)`, i.e. `dc/dx = -q`.
- Also: `compute_boundary_values` returns gradients along `+x`, not along the
  outward normal, so at the left end its gradient has the opposite sign to the
  one in the bc dictionary.
- Do: write the physical equation in `x` next to every dictionary, then derive
  `a` from it. Danckwerts inlet at `x = 0`: `v c - D dc/dx = v c_in` gives
  `{"a": D, "b": v, "d": v * c_in}`. A flux `N_in` entering at `x = L`:
  `-D dc/dx = -N_in` gives `{"a": D, "b": 0, "d": N_in}`.
- New: `print(pymrm.describe_bc(bc, x_f))` writes each dictionary out as the
  equation it imposes, in terms of `x`. Compare it with your physical equation.

## P2. Harmonic face mean at a diffusivity jump

The diffusion operator needs diffusivities at FACES. Where the diffusivity
differs between neighbouring cells, the face value is the harmonic mean
`2 / (1/D_i + 1/D_{i+1})` (for unequal cell sizes, weight by the half-cell
distances).

- Symptom: grid refinement converges at first order instead of second.
- Measured: composite slab, `D` drops tenfold at the midplane. Flux error with
  the arithmetic mean: 7.2 % at n = 10, 1.7 % at n = 40, 0.42 % at n = 160
  (order 1.00). Harmonic mean: below 1e-10 at every n.

## P3. `NumJac((n,))` builds a dense Jacobian

The default stencil couples the LAST axis in full. With a bare `(n,)` shape the
last axis is the spatial one, so every cell is declared coupled to every other.
That is right for a small pointwise system (`(n_c,)` species at one point) and
wrong for a field on a grid.

- Symptom: slow construction and solves; answers are unaffected.
- Measured: n = 400 takes 4.5 s to construct with nnz = n^2, against 0.0007 s
  and nnz = n for `(n, 1)`.
- New: a `UserWarning` for a 1-D shape with n >= 100 and the default stencil;
  `axes_blocks=[-1]` confirms that dense coupling is intended.
- Do: keep a field axis, `(n, 1)` for one field on a grid.

## P4. `axes_diagonals=[0]` on a 1-D shape

- Old (2.3.1 and older): WRONG Jacobian. `axes_blocks` defaults to `[-1]`, the
  same axis, so the `[-1, 0, 1]` offsets are read as absolute indices; for
  pointwise `-c**2` at `c = 1`, rows 2 onward have `-2` in column 0 and zero on
  the diagonal. Newton converges slowly or to a different answer.
- New: fixed; it gives the tridiagonal stencil, exact against finite
  differences.
- Do in any version: use `axes_diagonals` only when the SOURCE TERM itself
  reads neighbouring cells, preferably on a shape with a field axis,
  `NumJac((n, 1), axes_diagonals=[0])`. Diffusion and convection couplings
  enter analytically through the operators and need no stencil.

## P5. Changing boundary values: assemble once, use `shapes_d`

`construct_grad` and `construct_convflux_upwind` accept `shapes_d`. They then
return the boundary contribution as matrices that multiply an external vector
of boundary values, so a new boundary value costs one matrix-vector product.

- With `shapes_d` the dictionary's `d` is a COEFFICIENT on that external vector,
  not a value. Set `d = 1` and pass the values through the vector; with `d = 2`
  and a vector entry of 2 the boundary term is 4 (measured).
- Measured: with `d = 1`, `grad_bc_left @ [d_new]` equals the boundary vector of
  an operator rebuilt with `d_new`, and the operator matrix itself is unchanged.
- Do: `grad, grad_bc_left, grad_bc_right = construct_grad(shape, x_f, x_c, bc,
  shapes_d=((1,), None))` with `d = 1` in the left dictionary, then use
  `grad_bc_left @ d` in the residual. The shape in `shapes_d` must broadcast to
  the boundary's shape (the field shape with 1 on the axis).

## P6. Pure outflow from a stirred volume

A zero-gradient outlet `{"a": 1, "b": 0, "d": 0}` makes the upwind operator
reconstruct the exit face value from the last cells. That is correct for a
discretised PDE and wrong when the last cell is a stirred volume whose exit
carries its own value (tanks in series, a cell model, a CSTR cascade).

- Measured: first-order reaction in N equal tanks, `k tau = 1`. Zero-gradient
  outlet error: 3.85 % at N = 2, 1.25 % at N = 8.
- New: `{"outflow": True}` as the outlet dictionary; the face value is the
  adjacent cell value, diffusion sees zero gradient there. Exact.
- Old (exact workaround): put NO flux on the exit face (`{"a": 0, "b": 1,
  "d": 0}`) and add the outflow `v / dz_last` as a sink on the last cell with
  `construct_coefficient_matrix`.

## P7. `newton` stops on an ABSOLUTE step

`newton(..., tol=1.49e-8)` stops when the infinity norm of the Newton step drops
below `tol`, whatever the magnitude of the unknowns.

- Symptom: a small unknown (mole fractions of a trace species, concentrations in
  kmol, coverages near zero) "converges" at once to a wrong value with
  `success=True`. A large unknown (pressure in Pa) may never meet the tolerance.
- Measured: solving `x^2 = 1e-20` from `x0 = 1e-8` returns `success=True` after
  one iteration with `x = 5e-9`, 50 times the true root.
- Do in any version: scale every unknown to order one before solving; with mixed
  units scale each field separately.
- New: `newton(..., tol=0.0, rtol=1e-10)` stops on a relative step and finds the
  root above to 1e-8. `pymrm.checks.residual_check(fun, x)` judges a solution by
  a scale-free backward error (0.33 for the wrong root above, 4e-17 for the
  right one). An absolute residual threshold is not scale-free: it rejected
  converged solves in other units in agent runs.

## P8. Read outlet and wall values from the face

The operators transport the reconstructed FACE value. A hand-written outlet flux
`v * c[-1]` (last cell centre) is not what the model transported.

- Symptom: a mass or energy balance that fails to close on a coarse grid and
  looks like a physics error.
- Measured: dispersed plug flow with first-order reaction, n = 8. Balance with
  the face value from `compute_boundary_values`: 2e-16. With `v * c[-1]`:
  3.8e-3. The gallery once published an outlet metric 11.4 % low for reading half
  a cell short of the boundary.
- Do: `c_in_face, _, c_out_face, _ = compute_boundary_values(c, x_f, x_c, bc)`.
