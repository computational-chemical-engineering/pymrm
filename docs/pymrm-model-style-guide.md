# PyMRM Model Style Guide

This guide is intended for `pymrm` exercise solutions and class demonstrations.
It is based on the current `pymrm` tutorials and API, the extended teacher
solutions in `pymrm-book-teacher`, and the implementation patterns already used
in `exercises/solutions`.

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

Two formats are allowed.

### 2.1 Compact script format

Use this for:
- L1-L2 ODE examples
- single-purpose demonstrations
- short derivations where the numerical method itself is the teaching target

Required section order in a `.py` file or notebook:

1. Problem statement
2. Imports
3. Parameters
4. Model equations in code
5. Discretisation / solver loop
6. Post-processing
7. Short validation note

### 2.2 Class-based format

Use this by default for:
- all PDE models
- multicomponent models
- multiphase models
- reusable demos
- any model with more than one solve mode or more than one physical field

This is the preferred style from L3 onward.

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

Notebooks should render clearly in three places:

1. A running Jupyter notebook or JupyterLab session.
2. The GitHub notebook preview.
3. The VS Code notebook editor.

Use portable notebook Markdown by default:

- Use ordinary Markdown headings, lists, links, tables, fenced code blocks, and
  LaTeX equations.
- Use display equations with `$$ ... $$` and inline equations with `$...$`.
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

- 1D single field: `(n_x,)`
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
a * normal_gradient + b * value = d
```

Rules:

1. Define `bc` immediately after the grid and before operator assembly.
2. Add a short comment showing the physical meaning of each boundary.
3. For multicomponent or multiphase problems, shape `a`, `b`, and `d` so that
   their array structure matches the non-spatial axes.
4. When using Robin or Danckwerts conditions, write the physical equation in a
   nearby comment or Markdown cell before the dictionary.

Example:

```python
# Danckwerts inlet: D * dc/dx + v * c = v * c_in
bc = (
    {"a": d_ax, "b": velocity, "d": velocity * c_in},
    {"a": 1.0, "b": 0.0, "d": 0.0},  # zero outlet gradient
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

### 9.2 Convection

Use:

```python
conv_mat, conv_bc = construct_convflux_upwind(shape, x_f, x_c, bc, v=v, axis=axis)
```

If TVD is used, keep the first-order upwind part in the constant Jacobian and
add the limiter correction separately.

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

This is especially useful in L1-L2, where the stoichiometric structure is part
of the teaching objective.

## 11. Solver Rules

1. Use `newton(...)` for nonlinear implicit steps and steady-state solves.
2. Use `NumJac(...)` when an analytical Jacobian is not trivial.
3. Use `clip_approach(...)` when positivity or boundedness is physically
   required.
4. For linear steady problems, use `spsolve(...)` directly.
5. Do not rebuild constant sparse matrices inside the time loop.

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

Examples:

- diffusion-only with homogeneous boundaries gives a flat profile
- first-order diffusion-reaction matches the analytical hyperbolic solution
- TVD scheme is sharper than FOU but remains bounded
- total mass is conserved in a closed reaction system
- concentration stays non-negative

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
            {"a": 1.0, "b": 0.0, "d": 1.0},
            {"a": 1.0, "b": 0.0, "d": 0.0},
        )

        # State
        self.u = np.zeros((self.n_x,))

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

If a new exercise or demo is written today, the preferred house style is:

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

This should be the default standard for future `pymrm` exercise solutions and
class demonstrations.
