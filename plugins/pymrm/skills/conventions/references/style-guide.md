# PyMRM Model Style Guide

This guide is the house standard for `pymrm` models: exercise solutions, class
demonstrations, research models, and models written by coding agents. It is based
on the current `pymrm` tutorials and API and on the patterns used across the
`pymrm` examples and the pymrm gallery.

Read it together with [`pitfalls.md`](pitfalls.md): API behaviour that has
produced wrong answers in practice, each entry backed by a test in
`test/test_pitfalls.py` of the pymrm repository.

A measured example is cited by its gallery catalogue ID (`B1.1`) and the row it
came from as "(gl N)": row N of `docs/refresh-learnings.md` in the pymrm gallery
(<https://github.com/computational-chemical-engineering/pymrm-gallery>).

The goal is not to make every model identical. The goal is to make them
predictable: same structure, same naming, same array conventions, and same
separation between physics, discretisation, solver, and plotting.

## 1. Design Principles

1. Prefer `pymrm` operators over hand-written finite-difference stencils.
2. Keep the physics readable: define balances, source terms, and boundary
   conditions in code as directly as possible.
3. Separate constant linear operators from state-dependent nonlinear terms.
4. Use one consistent array layout across the whole model.
5. Make every model easy to validate against a limit case, analytical solution,
   or known trend.
6. Keep plotting and reporting outside the core residual whenever possible.

## 2. Recommended Output Formats

Choose the format by purpose and size, not by whether the model is a PDE:

| Case | Format |
|---|---|
| Didactic or simple: one domain, one or two fields, one solve mode, a teaching or readability profile | one flat, executed notebook, no class (2.1): operators built in visible steps, markdown between them, checks at the end |
| A full reactor model: several fields or domains, several solve modes, meant to be extended or reused | `.py` module(s) with the model (2.2), plus a notebook that drives the model and reports results |
| In between | one `.py` module plus a notebook |

State the choice in the specification; an explicit wish of the user always
wins. Notebooks are delivered executed, with outputs. If `nbformat`/`nbclient`
(or jupyter) are not available to execute one, write it anyway and say that it
was not executed; do not silently switch to a script.

The two formats:

### 2.1 Compact script or flat notebook format

Use this for:
- pointwise algebra and ODE models (batch, CSTR, ideal PFR marching)
- single-purpose demonstrations
- short derivations where the numerical method itself is the teaching target
- simple didactic PDE models (see `exemplars/pellet_teaching.ipynb`)

Required section order in a `.py` file or notebook:

1. Problem statement
2. Imports
3. Parameters
4. Model equations in code
5. Discretisation / solver loop
6. Post-processing
7. Short validation note

### 2.2 Class-based format

Use this for full reactor models as defined by the table above: several fields
or domains, several solve modes, or a model meant to be extended and reused
(multicomponent and multiphase models usually are).

## 3. Standard Section Order

For notebooks, use Markdown headings in this order:

1. `## Background`
2. `## Governing Equations`
3. `## Parameters and Assumptions`
4. `## PyMRM Implementation`
5. `## Results`
6. `## Validation`
7. `## Discussion`

For scripts, use comment banners in the same order.

### 3.1 Notebook formatting and figures

Notebooks should render clearly in four places:

1. A running Jupyter notebook or JupyterLab session (MathJax).
2. The GitHub notebook preview (Markdown is processed before the maths).
3. The VS Code notebook editor (KaTeX).
4. Google Colab.

Use portable notebook Markdown by default:

- Use ordinary Markdown headings, lists, links, tables, fenced code blocks, and
  LaTeX equations.
- Use display equations with `$$ ... $$` and inline equations with `$...$`,
  within this portable subset. Required (confirmed to break somewhere by
  `docs/math-render-test.ipynb`):
  - no `\label` (breaks VS Code) and no `\ref` or `\eqref`, also not in the text
    (no equation number on GitHub or in Colab); refer to equations in words;
  - at most one `\tag` per display block, and none in inline maths;
  - no `\bm` (JupyterLab's MathJax lacks it; use `\boldsymbol`);
  - every expression parses in KaTeX (the VS Code engine).
  Advisable (rendered in the test, or not covered by it):
  - multi-line equations with `aligned`, `cases` or `array` inside `$$` rather
    than a bare `\begin{align}`;
  - no `\\` and no pairs of `*` in inline maths (`\ast`, `\cdot`);
  - no macros (`\newcommand`, `\def`); they do not carry over between cells;
  - multi-letter subscripts in `\mathrm{}`;
  - no maths in headings; `\vert` rather than `|` inside maths in a table.
- Check a notebook with the conventions skill's
  `scripts/check_notebook_math.py notebook.ipynb`, which lints these rules and
  parses every expression with KaTeX when Node.js and KaTeX are installed. It
  reports the required rules as errors and the advisable ones as warnings.
- Avoid MyST-only directives such as `{figure}`, `{grid}`,
  admonitions, and colon-fenced layouts when the notebook must also be readable
  outside the Jupyter Book build.
- Avoid raw HTML unless there is no practical Markdown alternative.
- Keep one conceptual step per Markdown or code cell. Long cells with many
  unrelated equations, figures, and instructions are harder to review and more
  likely to render inconsistently.

Generated figures:

- Figures produced by Python plotting functions should be created by code cells
  when the notebook is run.
- Do not save generated plots to external image files just to include them in a
  later Markdown cell.
- Use the inline Matplotlib backend, create figures explicitly with
  `fig, ax = plt.subplots(...)`, label axes with symbol and unit, and call
  `plt.show()` when helpful for reliable rendering.
- If a generated figure is useful in the GitHub preview, keep the cell output in
  the notebook. The source of truth remains the plotting code, not an external
  image file.

Static or externally prepared figures:

- Embed static figures as notebook attachments, not as separate files in a
  `media/` directory, when the notebook is intended to be downloadable as a
  single self-contained `.ipynb` file.
- Reference attached figures with standard Markdown:

  ```markdown
  ![My figure](attachment:my_figure.png)
  ```

- Use short lowercase file names without spaces, for example
  `bubble_column_schematic.png`.
- Prefer PNG for raster figures and SVG only when it is known to render
  correctly in Jupyter, GitHub, VS Code, and the Jupyter Book build.
- Keep a short italic caption directly below the image when a caption is needed,
  because this renders consistently in ordinary notebook Markdown:

  ```markdown
  ![Bubble-column schematic](attachment:bubble_column_schematic.png)

  *Schematic representation of the bubble-column model.*
  ```

## 4. Naming Conventions

Use descriptive lowercase names for parameters and lowercase-with-underscores
for counts and coordinates.

### 4.1 Scalars

- `length`, `radius`, `dt`, `dz`, `dr`
- `velocity`, `d_ax`, `d_r`, `k_rxn`, `k_ext`, `u_wall`
- `t_end`, `maxfev`, `tol`

Short symbols are allowed only when they are standard and local:

- `v`, `D`, `k`, `T`, `R`

### 4.2 Grid and count variables

- `n_x`, `n_z`, `n_r`, `n_c`, `n_phase`
- `x_f`, `x_c`, `z_f`, `z_c`, `r_f`, `r_c`

Rule:
- suffix `_f` means face locations
- suffix `_c` means cell-center locations
- prefix `n_` means number of cells or fields

### 4.3 State variables

- `c` for concentration-only models
- `T` for temperature-only models
- `u` for combined state vectors, such as `[c, T]` or multi-field states
- `c_old`, `u_old` for previous-step states

### 4.4 Operator names

- `grad_mat`, `grad_bc`
- `conv_mat`, `conv_bc`
- `div_mat`
- `flux_mat`, `flux_bc`
- `jac_diff`, `jac_conv`, `jac_react`, `jac_accum`, `jac_const`
- `g_const`
- `numjac`

Do not mix names such as `Grad`, `Flux`, `Jac_const`, `construct_Jac`,
`init_Jac`, and `jac_const` in the same code base. Standardise on lowercase.

### 4.5 Residual naming

Prefer `residual(...)` over `g(...)`.

Use:

- `residual(...)` for the nonlinear algebraic balance
- `jac` for the Jacobian returned together with the residual
- `g_const` for the constant contribution to the residual

Rationale:

1. `residual` is immediately clear to students and readers.
2. `g` is compact but not descriptive enough for teaching material.
3. Keeping `g_const` is still fine because it is an internal implementation
   detail rather than the public interface of the model.

## 5. Array Shape Conventions

Spatial axes come first. Non-spatial axes come last.

Preferred layouts:

- 1D single field: `(n_x, 1)`, never `(n_x,)` (see pitfalls P3)
- 1D multicomponent: `(n_x, n_c)`
- 1D two-phase single-component: `(n_z, n_phase, 1)`
- 1D two-phase multicomponent: `(n_z, n_phase, n_c)`
- 2D multicomponent: `(n_z, n_r, n_c)` or `(n_x, n_y, n_c)`
- coupled concentration-temperature field: `(n_r, 2)` or `(n_z, n_r, 2)`

Rules:

1. Use the same layout everywhere in one model.
2. Pass the correct `axis` explicitly to `pymrm` operators.
3. Keep components and phases in the final axes so broadcasting remains clear.
4. When flattening for linear algebra, only flatten at the residual/Jacobian
   interface.
5. Keep a field axis even when there is one field. `NumJac` couples the last
   axis in full, so a bare `(n_x,)` shape builds a dense `n_x` by `n_x` Jacobian.

## 6. Boundary Conditions

Always use the `pymrm` convention:

```python
bc = (
    {"a": ..., "b": ..., "d": ...},  # left / lower boundary
    {"a": ..., "b": ..., "d": ...},  # right / upper boundary
)
```

Interpretation:

```text
a * dc/dn + b * c = d      with n the OUTWARD normal
```

At the right (upper) boundary `dc/dn = dc/dx`; at the left (lower) boundary
`dc/dn = -dc/dx`. The same dictionary therefore means opposite fluxes at the two
ends. `compute_boundary_values` returns gradients along `+x`, not along the
outward normal. See pitfalls P1.

Rules:

1. Define `bc` immediately after the grid and before operator assembly.
2. Add a short comment showing the physical meaning of each boundary.
3. For multicomponent or multiphase problems, shape `a`, `b`, and `d` so that
   their array structure matches the non-spatial axes.
4. When using Robin or Danckwerts conditions, write the physical equation in a
   nearby comment or Markdown cell before the dictionary.
5. Write that comment in terms of `x`, and check the sign of `a` against the
   outward normal at that end.
6. When a boundary value changes during the run (time-varying feed, continuation
   on an inlet value), assemble once with `shapes_d` and multiply the returned
   boundary matrices by the new `d`; do not rebuild the operators.

Example:

```python
# Danckwerts inlet at x = 0: v * c - D * dc/dx = v * c_in
# (outward normal is -x, so -D * dc/dx = +D * dc/dn and a = +D)
bc = (
    {"a": d_ax, "b": velocity, "d": velocity * c_in},
    {"a": 1.0, "b": 0.0, "d": 0.0},  # outlet at x = L: dc/dx = 0
)
```

## 7. Preferred Class Structure

Use this method order:

```python
class ModelName:
    def __init__(self, ...):
        ...

    def _build_grid(self):
        ...

    def _init_state(self, ...):
        ...

    def _build_operators(self):
        ...

    def reaction(self, u):
        ...

    def residual(self, u, u_old=None):
        ...

    def solve(self, ...):
        ...

    def postprocess(self):
        ...
```

Notes:

- Use `_build_grid` and `_build_operators` for one-time setup.
- Use `reaction(...)` or `source(...)` only for state-dependent physics.
- Use `residual(...)` as the single source of truth for Newton solves.
- Treat `g(...)` as legacy naming only when preserving older material.
- Keep `solve(...)` short. It should orchestrate, not define physics.

## 8. Preferred Residual Pattern

For implicit time stepping:

```python
def residual(self, u, u_old):
    g_rxn, jac_rxn = self.numjac(self.reaction, u)
    g = (
        self.g_const
        + self.jac_const @ u.reshape((-1, 1))
        - u_old.reshape((-1, 1)) / self.dt
        - g_rxn.reshape((-1, 1))
    )
    jac = self.jac_const - jac_rxn
    return g, jac
```

For steady problems:

```python
def residual(self, u):
    g_rxn, jac_rxn = self.numjac(self.reaction, u)
    g = self.g_const + self.jac_const @ u.reshape((-1, 1)) - g_rxn.reshape((-1, 1))
    jac = self.jac_const - jac_rxn
    return g, jac
```

Rules:

1. Return both residual and Jacobian.
2. Keep the sign convention consistent across the whole model.
3. Store constant linear pieces in `self.g_const` and `self.jac_const`.
4. Keep nonlinear source terms outside `self.jac_const`.

## 9. Operator Assembly Rules

### 9.1 Diffusion

Use:

```python
grad_mat, grad_bc = construct_grad(shape, x_f, x_c, bc, axis=axis)
div_mat = construct_div(shape, x_f, nu=nu, axis=axis)
flux_mat = -diff_mat @ grad_mat
flux_bc = -diff_mat @ grad_bc
```

`diff_mat` holds FACE diffusivities. Where the diffusivity varies between cells,
use the harmonic mean of the two neighbours at each interior face; the
arithmetic mean converges at first order at a jump (pitfalls P2).

### 9.2 Convection

Use:

```python
conv_mat, conv_bc = construct_convflux_upwind(shape, x_f, x_c, bc, v=v, axis=axis)
```

If TVD is used in a steady problem or a Newton solve, use `ConvFluxTVD`: it
returns the limited fluxes and their exact Jacobian, and `newton` with
`damping="monotone"` converges for every limiter. In a time step, deferred
correction is also acceptable: the first-order upwind part in the constant
Jacobian, the limiter correction in the residual, one or two inner iterations
(pitfalls P10).

### 9.3 Accumulation

Use:

```python
jac_accum = eye_array(n_total, format="csc") / dt
```

or `construct_coefficient_matrix(...)` if the accumulation coefficient varies
per field or per cell.

### 9.4 Geometry

Use `nu` explicitly:

- `nu=0` Cartesian 1D
- `nu=1` cylindrical radial
- `nu=2` spherical radial

Do not rely on memory for this. Put the geometry in a comment next to the call.

`nu` is not always the only place the geometry enters. A Thiele modulus or a
Robin condition defined on `V_p/S_x` carries it again in `R/(V_p/S_x)` (3 sphere,
2 cylinder, 1 slab): changing only `nu` gives a model that converges and is wrong
(`G1.8`: the slab misses its closed form by 0.67, and agrees to 1.1e-5 with the
ratio changed too). Check every change of geometry against the closed form for
that geometry, and change it in the post-processing on the same basis (`E1.2`: a
2-D through-flow divided by a sphere's volume gave 1.032 where 4.5 is expected)
(gl 30, 154).

## 10. Nonlinear Source Terms

### 10.1 Preferred style

Define the physical source term separately:

```python
def reaction(self, c):
    r = self.k_rxn * c[..., 0] * c[..., 1]
    f = np.zeros_like(c)
    f[..., 0] = -r
    f[..., 1] = -r
    f[..., 2] =  r
    return f
```

### 10.2 Stoichiometric form

For ODE or reaction-network models, prefer:

```python
rates = ...
rhs = nu @ rates
```

This is especially useful in reaction-network and ODE models, where the stoichiometric structure is part
of the teaching objective.

### 10.3 Floating point in sources and post-processing

- Write closed forms that cancel near a parameter value without the cancellation:
  `np.expm1((d - 1) * np.log(tau)) / (d - 1)`, not `(tau**(d - 1) - 1) / (d - 1)` near
  `d = 1`. No solver tolerance recovers the lost digits (`B2.3`: bias 1.0e-6 became
  8.4e-9) (gl 12).
- A difference quotient of such a closed form reports the cancellation divided by the
  step. Before calling a gap truncation, repeat with the same step on a series and with
  a larger step: truncation shrinks with the step, round-off grows (`B1.4`: 1.2e-5
  published as truncation, 6.7e-8 on the series) (gl 109).
- Compute step times as `(i + 1) * dt`, not by `t += dt`, and select time windows with a
  half-step margin: ten steps of 0.01 sum to 0.09999999999999999, and `t >= 0.1` drops
  the edge (`B3.2`: the dropped edge held the maximum error, 6.2e-5 against the reported
  4.3e-5) (gl 133).

## 11. Solver Rules

1. Use `newton(...)` for nonlinear implicit steps and steady-state solves.
2. Use `NumJac(...)` when an analytical Jacobian is not trivial. Give it the
   field-shaped state, `(n_x, 1)` for one field. Use `axes_diagonals` only on
   shapes with at least two axes, and only when the source term itself reads
   neighbouring cells (pitfalls P3, P4).
3. Write rates that are smooth and first order through zero concentration
   (`k c (c**2 + eps**2)**((n - 1)/2)` instead of `k c**n` with `n < 1`;
   no `np.maximum(c, 0)`), so that a slightly negative iterate gives a
   restoring rate (pitfalls P9). Use `newton(..., bounds=...)` (from 3.0.0;
   `clip_approach(...)` as the callback before) only for unknowns the residual
   cannot accept beyond a limit: `T > 0`, arguments of `log` or `sqrt`.
4. For linear steady problems, use `spsolve(...)` directly.
5. Do not rebuild constant sparse matrices inside the time loop.
6. `newton` tests the infinity norm of the STEP against `tol`, an absolute
   number. Scale unknowns so that they are of order one (or pass `scale=`).
   Up to 2.5.0 a trace-level unknown reports success after one step with a
   wrong value; from 3.0.0 the default `backward_tol=1e-8` keeps iterating
   (pitfalls P7). Check `result.success`; `result.status` and
   `result.message` say why a solve stopped.
7. When a solve fails from a reasonable guess, follow the order in the API
   map's "Hard nonlinear problems": `damping="guarded"` (the default after
   2.5.0), then regularised kinetics with `damping="monotone"`, `scale=`,
   then `pseudo_transient` or `continuation`. Do not wrap `newton` in a
   home-made line search.
8. Newton knows nothing of dynamic stability: from a nearby guess it converges
   to an unstable steady state as fast as to a stable one (`B1.1`: the middle
   branch in 4 iterations). Show stability by time or pseudo-time marching, and
   folds by continuation in the physical parameter. A pseudo-time march shows physical
   stability only with the physical mass matrix (for example, the Lewis number) (gl 18).
9. Locate a root (a threshold, a crossing, a length) with `find_roots` over a
   bracket that follows from the physics, and raise unless it finds exactly one
   (`E1.1`: `[L_mf, L_mf / (1 - delta_max)]` instead of hand constants). Where
   the feature can lie outside the domain, decide from a parameter test whether
   it exists, and raise when it exists but the domain is too short, instead of
   returning `nan` for both (`E1.2`) (gl 152, 156).

Preferred `solve()` pattern:

```python
def solve(self, n_steps, callback=None):
    for step in range(n_steps):
        u_old = self.u.copy()
        result = newton(lambda u: self.residual(u, u_old), self.u, maxfev=self.maxfev)
        self.u = result.x.reshape(self.u.shape)
        if callback is not None and step % self.output_interval == 0:
            callback(step, self)
```

## 12. Plotting and Post-Processing

Rules:

1. Keep plotting outside `reaction(...)` and `residual(...)`.
2. Use a callback or dedicated plotting method for live demonstrations.
3. Label every axis with both symbol and unit.
4. Do not save notebook-generated plots as external files for later inclusion in
   the same notebook. Generate them from code when the notebook is run.
5. If a plot should be visible in the GitHub preview, keep the executed output
   in the notebook.
6. For static figures that are not generated by code, use notebook attachments
   with `![My figure](attachment:my_figure.png)`.
7. Compute derived engineering quantities in named methods:
   `effectiveness_factor()`, `apparent_rate()`, `cup_mixing_average()`,
   `conversion()`, `selectivity()`.

## 13. Validation Requirements

Every model should include at least one of the following:

1. Analytical comparison
2. Limit-case comparison
3. Conservation check
4. Grid-independence check
5. Physical monotonicity or boundedness check

Every check must be able to fail. Break the model on purpose (flip a sign, change
`nu`, mismatch a bc) and confirm the checked number fails its own tolerance;
moving is not enough (13.1). Refine every axis that carries error (grid and time
step) and report the observed order (13.2). Read outlet and wall values with
`compute_boundary_values`, not from the last cell centre (pitfalls P8).

Examples:

- diffusion-only with homogeneous boundaries gives a flat profile
- first-order diffusion-reaction matches the analytical hyperbolic solution
- TVD scheme is sharper than FOU but remains bounded
- total mass is conserved in a closed reaction system
- concentration stays non-negative

### 13.1 Break rows

- A defect protects a check only if it fails that check's tolerance. Under a 5 %
  tolerance a 1 % defect is invisible (`J4.6`: `k2 + 1 %`). Size the defect against the
  tolerance, or tighten the tolerance, no further than about 100 times the measured
  spread; if the defect still passes, enlarge it or label the metric structural (gl 7).
- Measure a defect's effect as the difference between two solves that differ only by the
  defect, on the same grid. Against an external reference each solve adds its own
  discretisation error (`A2.8`: a far-field defect reported as 5.9e-7, the grid error,
  moves the answer by 2.5e-16) (gl 70).
- A comparison of an approximation with an exact solve needs the case where the
  approximation is exact by construction, and a unit check of the formula (`A3.1`: with
  a linear equilibrium the misprinted additivity rule missed the exact answer by 109 %)
  (gl 78).
- For a closed-form correlation, keep every published constant in one dictionary that
  every function takes as an argument; one defect per constant (digits transposed, sign
  flipped), and one that flips each threshold claim (`F1.3`: 42 defects failed 88 of 89
  claims) (gl 175).
- A scan whose cases are related by a symmetry of the model counts each case more than
  once. Scan the fundamental domain and check the symmetry once as an identity, with a
  defect that breaks it (`A2.5`: 12 rows were 8 beds, and the two "exceptions" one bed)
  (gl 65).

### 13.2 Observed orders

- Fit orders from successive differences of three levels at a constant ratio
  (`observed_orders`), and quote the error against the Richardson limit. Errors against
  the finest grid, or against the exact value on a ladder that refines one axis only,
  carry a fixed error in every level (`H1.9`: 2.39 against the finest grid, 2.09 and
  2.05 from differences; `A3.15`: 2.12 against 2.01) (gl 43, 75).
- Refine every axis before calling a remaining gap a difference. "The solver cannot
  resolve this effect" is about one grid, not the solver (`A3.2`: the gap left at a
  fixed step count, 0.18 %, was the step error) (gl 82).
- Report the order per case, not averaged, and check it at a second parameter value
  before explaining an excess. A thin reacting layer converges at orders between 2 and 3
  on practical grids: bound such an order by [2, 3] (from `B1.1` and `B1.2`), and by [1.9, 2.1] only where the
  layer is resolved (`B1.1`: mean 2.35 was 2.96, 2.06, 2.03; the slab gives 2.0 at phi =
  1). Check the arithmetic: a ratio 2.29 at a 1.33x refinement is order 2.9 (`B1.2`) (gl
  19, 95).
- Orders that oscillate (upwind on a shock: 0.88 to 1.12 on 50 to 1600 cells, `A1.5`)
  are reported as a minimum and a maximum. A pre-asymptotic order from nearly equal
  differences moves with the numerical stack (`J1.10`: 1.79, later 2.11): report a range
  with its reason (gl 9, 163).
- Show that the finest difference is above round-off: measure round-off by the residual
  of an identity the discrete equations satisfy exactly, such as a conservation gap
  (`B3.1`: finest difference 3.1e3 times the gap; 3.95 on a ladder taken too fine, which
  is the defect for this check) (gl 132).

### 13.3 Round-off and tolerance-level numbers

- Classify each checked number by what it is: zero by construction (an identity, checked
  at a bound of round-off), set by a solver or integrator tolerance, or physical. Never
  decide by its size, and never state a round-off value as a number: state the bound
  (`J4.6`: 9.7e-16 stated, 4.8e-16 on another machine) (gl 1, 2).
- An integrator meets a closed form to round-off when the solution is linear in the
  integrated variable; that is an identity, not a tolerance bound (`B2.3`: 2.7e-15
  against a bound of 1e-7) (gl 129).
- Never divide by a round-off quantity, also not in a printed sentence (`J4.8`: 7e-12 on
  one BLAS kernel, 0.0 on another, `ZeroDivisionError`) (gl 6).
- Do not count "exact zeros" after a Newton step: count on the construction that makes
  them, or below a threshold with a margin (`H1.1`: one step of 1e-16 moved 264 of 3014
  zeros) (gl 40).

### 13.4 Roots, limits and extremes

- An argmax on the last point of the domain says "at or beyond the end": extend the
  domain (`C2.10`: the maximum was at 6.65 m, not at the 3 m exit) (gl 22).
- A crossing read as the nearest node of a grid is the node: use `find_roots`, and for
  an extremum the roots of the derivative. A maximum over a range taken on a grid is
  often the value at the edge; say what limit it tends to (`J1.5`: 0.0912 on the grid,
  0.0940 as root) (gl 182).
- Take limits in closed form, and check on the grid that the approach is monotone
  (`A1.6`: the grid end gave 1.21x, the limit 1.17x). A limit evaluated at a finite
  point is a rate: check it as a bound at that point (`A1.1`: 1.5e-7 at x = 1e9).
  Evaluate it through the dimensionless function, at the dimensionless value you state
  (`A1.8`: velocities of 1e-9 m/s were Re 2e-8) (gl 147, 189, 221).
- A bounded minimiser of a flat, noisy objective resolves the minimum only to about
  `sqrt(2 dF / F'')`; root a cancellation-free derivative instead (`B3.3`: a one-ulp
  change of the bracket moved the result from 1.8e-9 to 1.2e-7) (gl 138).

### 13.5 Statements in prose

- Generate every number and count in the text from the code or the loaded table that
  holds it; never retype one (`D1.1`: text said five, the code printed 6; `F1.3`: "ten
  times" was 1.25 / 0.16 = 7.8). Take another model's results from its stored output and
  its code unchanged, by import or byte-identical copy (`A1.8`: against retyped
  constants 2.6e-8, against the stored values 0) (gl 55, 174, 213, 225).
- A failure mode or trap stated in the text needs the code that produces it (`D1.1`: a
  quoted 123 K error has no code path, the solve gives 0.52 K; `A2.1`) (gl 53, 60).
- Compare with a bound as the ratio to the bound, with its margin; compute a closeness
  ("within a few percent") over the whole stated range; and measure a negligibility on
  the data the model is about, not at a few chosen conditions (`I1.3`: "2.0x" was 2.3 to
  7.7; `A1.1`: within 10 % only for 7.8 < x < 28.6; `F1.3`: 15 % in a table, 138 % on
  the data). A physical bound needs a number the model holds (`A1.8`) (gl 52, 146, 170,
  224).
- Compute a sensitivity of a coupled system by solving it again with the parameter
  changed, and say what is held fixed (`G1.7`: a quoted 44 % was 21 to 41 % through the
  loop). "A fit absorbs this effect" needs the equivalent parameter root-found per case
  (`E2.1`: 5.00 to 3.85 cm, not one 4.2 cm) (gl 29, 168).
- A constant that matches at one point does not show that the assumption behind it holds
  there: measure the assumption (`J1.5`: the LDF coefficient equals 15 where the profile
  is far from parabolic) (gl 180).
- Say what two "independent" routes share (`E1.1`: a closed form and an ODE route shared
  the end condition and the source function) (gl 150).
- Measure a grid error at every point a comparison uses, not where the profile is
  flattest (`C2.1`: 2.1e-5 at one point, 4.1e-5 over 61) (gl 143).

### 13.6 Data and statistics

- Read printed precision from the string: `pd.read_csv(..., dtype=str)` beside the
  numeric column, since `0.270` as a float is `0.27` (`H1.8`: a rounding band of 8.43 %
  from the float, 0.92 % from the string). A verdict decided by rounding carries its
  distance to the rounding boundary, in last-digit units, and a defect that makes a
  "none" false; a precision claim over a set is a count, with the unit of each phrase
  printed (`B2.1`, `A2.6`: 5 of 7) (gl 46, 143, 206, 229).
- Bootstrap with enough resamples that the seed hardly moves the result: vectorise
  (`rng.integers` on group sums), take the tolerance from the spread over about a dozen
  seeds, check each verdict ("excludes 0") as its own claim, and give added draws their
  own generator (`A1.5`: with 2000 resamples an end moved by up to 0.03, or 15 %; with
  200000 by 0.003) (gl 157, 159).
- Fit an exponent or a coefficient with its standard error, pool groups with separate
  intercepts, and add a suspected confound as a regressor before naming it (`B2.1`: 2.9
  standard errors from zero became 1.0). Rank rival models by the paired standard error
  of each gap, and nested models by the extra sum of squares F test (`C1.1`: a
  "discriminating" gap was 0.63 SE) (gl 203, 267).

## 14. Anti-Patterns to Avoid

Avoid these patterns even if they appear in older exercises.

1. Hidden dependence on globals inside physics functions
   Example: `reaction(c)` reading `self.c` or `nx` from outer scope.
2. Mixed naming styles in one model
   Example: `Jac_const`, `flux_bc`, `construct_Jac`.
3. Recomputing constant matrices inside every Newton iteration or time step.
4. Hand-building sparse stencils when `construct_grad`, `construct_div`, or
   `construct_convflux_upwind` already express the same model.
5. Mixing plotting code into the residual or source-term functions.
6. Ambiguous state layout
   Example: changing from `(n_x, n_c)` to `(n_c, n_x)` mid-notebook.
7. Boundary-condition comments that do not match the `{"a","b","d"}`
   dictionary.

## 15. Recommended Minimal Template

```python
import numpy as np
from scipy.sparse import eye_array
from pymrm import construct_grad, construct_div, NumJac, newton


class ModelName:
    def __init__(self):
        # Parameters
        self.length = 1.0
        self.n_x = 100
        self.d_eff = 1.0
        self.k_rxn = 1.0
        self.dt = 0.01
        self.maxfev = 10
        self.output_interval = 10

        # Grid
        self.x_f = np.linspace(0.0, self.length, self.n_x + 1)
        self.x_c = 0.5 * (self.x_f[:-1] + self.x_f[1:])

        # Boundary conditions
        self.bc = (
            {"a": 0.0, "b": 1.0, "d": 1.0},  # x = 0: c = 1
            {"a": 1.0, "b": 0.0, "d": 0.0},  # x = L: dc/dx = 0
        )

        # State: one field, so shape (n_x, 1)
        self.u = np.zeros((self.n_x, 1))

        # Operators
        self._build_operators()

    def _build_operators(self):
        grad_mat, grad_bc = construct_grad(self.u.shape, self.x_f, self.x_c, self.bc)
        div_mat = construct_div(self.u.shape, self.x_f, nu=0)

        flux_mat = -self.d_eff * grad_mat
        flux_bc = -self.d_eff * grad_bc

        jac_diff = div_mat @ flux_mat
        g_diff_bc = div_mat @ flux_bc
        jac_accum = eye_array(self.n_x, format="csc") / self.dt

        self.g_const = g_diff_bc
        self.jac_const = jac_accum + jac_diff
        self.numjac = NumJac(self.u.shape)

    def reaction(self, u):
        return -self.k_rxn * u

    def residual(self, u, u_old):
        g_rxn, jac_rxn = self.numjac(self.reaction, u)
        g = (
            self.g_const
            + self.jac_const @ u.reshape((-1, 1))
            - u_old.reshape((-1, 1)) / self.dt
            - g_rxn.reshape((-1, 1))
        )
        jac = self.jac_const - jac_rxn
        return g, jac

    def solve(self, n_steps):
        for _ in range(n_steps):
            u_old = self.u.copy()
            result = newton(lambda u: self.residual(u, u_old), self.u, maxfev=self.maxfev)
            self.u = result.x.reshape(self.u.shape)
```

## 16. Recommended House Style for Future Material

If a new model, exercise or demo is written today, the preferred house style is:

1. Notebook first cell states the model, assumptions, and target quantity.
2. Code uses the class-based pattern for all PDE or multivariable models.
3. Array layout keeps spatial axes first and fields last.
4. Boundary conditions are always written in `{"a","b","d"}` form with a
   matching physical equation.
5. Constant operators are assembled once.
6. Nonlinear terms live in `reaction(...)` or `source(...)`.
7. Residuals are exposed through `residual(...)`.
8. Plotting is separate from solving.
9. At least one validation step is shown.

This should be the default standard for future `pymrm` models, exercise solutions and
class demonstrations.
