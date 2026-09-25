# pymrm pitfalls

API behaviour that has produced wrong answers in real models. Every entry is
demonstrated by a test in `test/test_pitfalls.py` of the pymrm repository, which
pins the numbers quoted here. If one of those tests starts failing, the API has
changed and this file must change with it.

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

- Symptom: slow construction and solves; answers are unaffected.
- Measured: n = 400 takes 4.5 s to construct with nnz = n^2, against 0.0007 s
  and nnz = n for `(n, 1)`. The gallery measured 70 s at n = 1600 and whole
  models 6x slower.
- Do: always keep a field axis, `(n, 1)` for one field.

## P4. `axes_diagonals=[0]` on a 1-D shape gives a wrong Jacobian

On a 1-D shape `axes_blocks` still defaults to `[-1]`, which is the same axis,
so the `[-1, 0, 1]` offsets are read as absolute indices. Most rows lose their
diagonal and get an entry in column 0.

- Symptom: Newton converges slowly or to a different answer.
- Measured: for pointwise `-c**2` at `c = 1`, rows 2 onward have `-2` in column
  0 and zero on the diagonal.
- Do: use `axes_diagonals` only with at least two axes, e.g.
  `NumJac((n, 1), axes_diagonals=[0])`, and only when the SOURCE TERM itself reads
  neighbouring cells. Diffusion and convection couplings enter analytically
  through the operators and need no stencil.

## P5. Changing boundary values: assemble once, use `shapes_d`

`construct_grad` and `construct_convflux_upwind` accept `shapes_d`. They then
return the boundary contribution as matrices that multiply `d`, so a new
boundary value costs one matrix-vector product.

- Symptom of not doing it: operators rebuilt inside a time loop or a
  continuation, slow, and easy to rebuild inconsistently.
- Measured: `grad_bc_left @ [d_new]` equals the boundary vector of an operator
  rebuilt with `d_new`, and the operator matrix itself does not change.
- Do: `grad, grad_bc_left, grad_bc_right = construct_grad(shape, x_f, x_c, bc,
  shapes_d=((1,), None))`, then use `grad_bc_left @ d` in the residual.

## P6. There is no pure-outflow boundary condition

A zero-gradient outlet `{"a": 1, "b": 0, "d": 0}` makes the upwind operator
extrapolate the exit face value from the last cells. That is correct for a
discretised PDE and wrong when the last cell is a stirred volume whose exit
carries its own value (tanks in series, a cell model, a CSTR cascade).

- Measured: first-order reaction in N equal tanks, `k tau = 1`. Zero-gradient
  outlet error: 3.85 % at N = 2, 1.25 % at N = 8.
- Do (exact): put NO flux on the exit face (`{"a": 0, "b": 1, "d": 0}`) and add
  the outflow `v / dz_last` as a sink on the last cell with
  `construct_coefficient_matrix`.

## P7. `newton` stops on an ABSOLUTE step

`newton(..., tol=1.49e-8)` stops when the infinity norm of the Newton step drops
below `tol`, whatever the magnitude of the unknowns.

- Symptom: a small unknown (mole fractions of a trace species, concentrations in
  kmol, coverages near zero) "converges" at once to a wrong value with
  `success=True`. A large unknown (pressure in Pa) may never meet the tolerance.
- Measured: solving `x^2 = 1e-20` from `x0 = 1e-8` returns `success=True` after
  one iteration with `x = 5e-9`, 50 times the true root. Scaling `x = 1e-10 y`
  gives the root to 1e-8.
- Do: scale every unknown to order one before solving; with mixed units (K,
  mol/m3, Pa) scale each field separately. Check the final residual, not only
  `success`.

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
