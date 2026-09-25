# From process to model structure

Structure codes (from the pymrm gallery taxonomy) name the mathematical shape of
a model and the pymrm ingredients it needs. A model can carry several codes.
Pick the codes, then the nearest exemplar.

Bundled exemplars live in the `conventions` skill under `exemplars/`. Gallery
pages are at
<https://github.com/computational-chemical-engineering/pymrm-gallery/tree/main/pages>;
use them for physics and checks, and prefer the conventions where they differ.

| Code | Structure | Typical processes | pymrm ingredients | Start from |
|---|---|---|---|---|
| S1 | ODE or algebraic in time or at a point | batch reactor, CSTR, tanks in series, kinetics | `newton` on scaled unknowns, backward Euler, or `solve_ivp` | `cstr_steady_states.py`; gallery A2.4, C1.1 |
| S2 | ODE marching in space | ideal PFR, plug-flow bed | marching loop with `newton` per step, or S4 at steady state | gallery D1.1 (rung 1) |
| S3 | 1-D steady boundary-value problem | catalyst pellet, film, membrane | `construct_grad`, `construct_div(nu)`, `NumJac`, `newton` | `steady_pellet.py`; gallery B1.4, B1.7 |
| S4 | 1-D transient PDE | dispersed plug flow, breakthrough, start-up | S3 plus accumulation, time loop, constant Jacobian factorised once | `dispersion_reactor.py`; gallery B2.2 |
| S5 | convection-dominated, sharp fronts | chromatography, adsorption fronts, coking fronts | `construct_convflux_upwind`, `interp_cntr_to_stagg_tvd` deferred correction | `api-map.md` (TVD); gallery A2.8, B2.2 |
| S6 | 2-D PDE | cooled tube with radial gradients, Graetz problem, monolith channel | per-axis operators on `(n_z, n_r, n_c)`, `nu=1` radially | gallery A2.3, I1.3, D1.1 (rung 5) |
| S7 | several phases or domains coupled | bubbling bed, membrane reactor, gas-liquid column | layout `(n_z, n_phase, n_c)`, `update_array_indices`, `NumJac(axes_blocks=[-2, -1])` | gallery E2.1, H1.4 |
| S8 | nested scales | reactor with resolved particles | Schur complement of the particle problem, `shapes_d` boundary unknowns | gallery D1.1 (rung 4), J3.5 |
| S9 | implicit multicomponent flux | Maxwell-Stefan, dusty gas | per-face linear solve, `NumJac(axes_blocks=[-1])` | gallery A4.2, A4.3 |
| S10 | differential-algebraic, constraints | electroneutrality, pressure-velocity, equilibrium | monolithic Jacobian with algebraic rows | gallery J3.1, F3.5 |
| S11 | population balance | crystallisation, particle size | size as an extra axis, growth as convection (S5) | none bundled |
| S12 | moving boundary | shrinking core, dissolving particle | front tracking or fixed grid with an indicator | gallery B3.1 |
| S13 | non-standard geometry | tapered channel, arbitrary area profile | `construct_div(nu=callable)` | gallery B1.2 |

If no row fits, say so, describe the structure in words, and ask the user
before inventing a new method.
