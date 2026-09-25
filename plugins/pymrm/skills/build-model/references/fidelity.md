# Scope and fidelity: which phenomena the model needs

The right model is the simplest one whose answer would not change the user's
decision if a neglected phenomenon were added. Decide that with numbers, not
with habit.

## Procedure

1. List candidate phenomena (table below). Include the ones you expect to be
   negligible; showing that they are is part of the result.
2. For each, estimate the deciding group with the user's numbers. Every input
   carries a label: `user`, `literature: <ref>`, `correlation: <name, validity
   range>`, `assumed`. Print the arithmetic.
3. Classify: negligible (criterion passed with margin), marginal (within a
   factor of 3 of the bound), important.
4. Choose the rung (below). Marginal phenomena go in, or go into a sensitivity
   run, never silently out.
5. For every excluded phenomenon write one line: what it would change and
   roughly by how much.
6. If a criterion cannot be evaluated because an input is unknown, say so and
   treat the phenomenon as marginal.

## A passed criterion bounds the RATE, not the decision

Every criterion below asks whether the local rate deviates by more than about
5 %. The user's decision may be far more sensitive than the rate: a runaway
threshold, a hot spot, conversion close to 100 %, a selectivity. In the pymrm
gallery's fixed-bed study (page D1.1), Mears' interphase criterion passed by a
factor 3.79, yet adding the film moved the root-found runaway threshold by 4.7 %,
more than four times the margin the design relied on. For sensitive outputs,
treat a passed criterion as "probably small" and compute the effect on the
output itself with the model.

## Deciding groups

Notation: `r_v` observed rate per unit PARTICLE volume (mol m-3 s-1); `r_bed`
rate per unit BED volume; `L_p` characteristic particle length: radius `R` for a
sphere or long cylinder, half-thickness for a slab or washcoat on an inert wall;
`V_p/S_p` particle volume over external surface (`R/3` sphere, `R/2` cylinder,
half-thickness for a slab); `c_s`, `c_b` surface and bulk concentration; `n`
reaction order; `D_eff` effective diffusivity; `k_c`, `h` film mass- and
heat-transfer coefficients; `lambda_e` particle effective conductivity; `E`
activation energy; `R_g` gas constant.

Shape constant `S` for internal gradients, from the first-order small-modulus
expansion `eta = 1 - 0.05 phi^2 / S + ...` (that is `1 - phi^2/3` slab,
`1 - phi^2/8` cylinder, `1 - phi^2/15` sphere), so the deviation is 5 % at
`phi^2 = S`: **slab 0.15, cylinder 0.4, sphere 0.75**. Use the one for the
actual shape; the sphere value is five times more lenient than the slab value.

The thresholds below mean "the rate deviates by less than about 5 %". Before
using a criterion from another source, check which volume its rate refers to
(particle, bed, catalyst mass): published forms differ by the bed voidage or
density.

| Phenomenon | Group | Negligible when | Notes |
|---|---|---|---|
| internal diffusion (observable) | Weisz-Prater `Phi = r_v L_p^2 / (D_eff c_s)` | `abs(n) Phi < S` | Mears (1971) prints `Phi < 1/abs(n)` for a sphere, which admits 6.3 to 6.8 % (measured, pymrm gallery B1.7). At `n = 0` the rate is unaffected until the reactant is depleted at the centre (sphere: `Phi > 6`). For exothermic reactions multiplicity can occur below `Phi = 1` (gallery B1.4): add the internal heat criterion. |
| internal diffusion (from kinetics) | Thiele `phi = L_p sqrt(k / D_eff)`, first order | `phi^2 < S` | `eta = tanh(phi)/phi` slab, `3/phi^2 (phi coth(phi) - 1)` sphere |
| external mass transfer | `abs(n) r_v (V_p/S_p) / (k_c c_b)` | `< 0.05` | film drop `c_b - c_s = r_v (V_p/S_p) / k_c`; for a sphere this is Mears' `r_v R abs(n) / (k_c c_b) < 0.15`, exact at first order (B1.7). `k_c` from a labelled correlation; `Sh = 2` is the conservative stagnant limit for a sphere |
| external heat transfer | `abs(dH) r_v (V_p/S_p) E / (h R_g T_b^2)` | `< 0.05` | same derivation with `T_s - T_b = abs(dH) r_v (V_p/S_p) / h`; sphere form `< 0.15` with `R` |
| internal heat | `abs(dH) r_v L_p^2 / (lambda_e T_s)` | `< S R_g T_s / E` | Anderson (1963) for a sphere (`S = 0.75`). Maximum internal rise (Prater): `abs(dH) D_eff c_s / lambda_e` |
| axial dispersion in a packed bed | `L/d_p` against `(20 abs(n) / Pe_p) ln(c_in/c_out)` | `L/d_p` larger | Mears (1971). `Pe_p = u d_p / D_ax` is about 2 for gases at `Re_p > 10` when `u` is the INTERSTITIAL velocity; much lower for liquids and at low `Re_p`. Use `u` on the same basis in the model and the correlation, and label the correlation |
| axial dispersion, general | `Pe_L = u L / D_ax` | compare the dispersion-model conversion with plug flow at the estimated `Pe_L` | cheap to compute exactly; see the `dispersion_reactor` exemplar |
| radial temperature gradients in a cooled tube | `abs(dH) r_bed R_t^2 / (lambda_er T_w)` | `< 0.4 (R_g T_w / E) / (1 + 4 lambda_er / (h_w R_t))` | from the area-mean rise `Q R_t^2 / (8 lambda_er) + Q R_t / (2 h_w)` with `Q = abs(dH) r_bed`; Mears (1971) writes the wall term `8 (R_p/R_t) / Bi_w` with `Bi_w = h_w d_p / lambda_er`, the same thing |
| gas-liquid reaction regime | Hatta `Ha = sqrt(k D_A) / k_L` (first order) | `Ha < 0.3`: reaction in the liquid bulk; `Ha > 3`: reaction in the film | for `Ha > 3` the enhancement is about `Ha` only while `Ha` is much smaller than the instantaneous limit `E_inf`; otherwise `E` tends to `E_inf`. For `Ha < 0.3`, compare `k eps_L` with `k_L a` to tell a kinetic bulk regime from a transfer-limited one. Between: solve the film model |
| thermal runaway, cooled bed | sensitivity of the hot spot to inlet or wall temperature | | root-find the critical value on the model; never read a threshold off a sampled curve |
| transient vs steady | process time scale against the slowest internal time scale (`L_p^2/D_eff`, `L/u`, thermal `rho c_p L_p^2 / lambda`) | ratio large: quasi-steady | |
| pressure drop | Ergun: `dp/dz` against `p` | `Delta p / p < 0.1` | beyond, couple the momentum balance |
| wall channelling | `d_t / d_p` | `> 10` | rule of thumb; below it, velocity profiles matter |

When the numbers are near a bound, compute both models. A sensitivity run
tells the user more than a criterion.

## The fidelity ladder

For a catalytic fixed bed (other processes follow the same logic):

1. ideal reactor: pseudo-homogeneous plug flow or CSTR, intrinsic kinetics;
2. plus axial dispersion;
3. plus interphase (film) resistance for mass and heat;
4. plus intraparticle diffusion and heat;
5. plus radial gradients (2-D);
6. plus dynamics, deactivation, pressure drop, as the question requires.

These are not always nested: a 2-D pseudo-homogeneous model (rung 5 without 3
and 4) is common and answers a different question from a 1-D heterogeneous one.

**Adding one phenomenon can move the answer away from the better model.** In the
pymrm gallery's fixed-bed study (page D1.1, root-found safe inlet partial
pressure, each shift measured against the rung-1 baseline): axial dispersion
+0.17 %, film -4.74 %, film plus particle +15.69 %, 2-D pseudo-homogeneous
-8.94 %. Measured like for like, resolving the particle behind the film moves
the answer +21.4 % from the film-only model, 4.5 times the film's own shift in
the other direction, so stopping at the film model leaves you further from the
particle model than plug flow was. Decide on the phenomena that pull in both
directions before stopping.

## Output (`scoping.md`)

```markdown
# Scoping: <short name>
| Phenomenon | Group | Value | Inputs (value, unit, label) | Bound (shape) | Verdict |
|---|---|---|---|---|---|
Chosen rung: <n>, because <reason tied to the decision>
Structure codes: <S..> (see structures.md)
Excluded, and what each would change: <one line each, with a magnitude>
Sensitive outputs checked on the model rather than by criterion: <list>
Could not evaluate: <criterion, missing input, treated as marginal>
```
