# Reactor type and model structure

Two different questions often arrive together: "which reactor suits this
process" (design) and "which model suits this reactor" (analysis). Keep them
apart. When the user already has a reactor, skip to the second table.

## Which reactor suits the process

Start from the chemistry and the phases, then heat, then catalyst life.

| Feature of the process | Tends toward | Why |
|---|---|---|
| positive-order kinetics, high conversion wanted | plug flow (tubular, packed bed) or batch | a stirred tank runs at outlet concentration; for order > 0 it needs more volume |
| series reactions, intermediate wanted (A -> B -> C) | plug flow or batch, stopped at the optimum | back-mixing lets B react further |
| parallel reactions, desired path of higher order | plug flow or batch, high concentration | high concentration favours the higher order |
| parallel reactions, desired path of lower order | stirred tank, or feed distributed along the reactor | low concentration favours the lower order |
| autocatalytic or substrate-inhibited kinetics | stirred tank followed by plug flow, or recycle | the stirred tank runs where the rate is highest |
| strongly exothermic, solid catalyst | multitubular cooled bed (narrow tubes), fluidised bed, or adiabatic stages with intercooling | removing heat is the design constraint; runaway risk |
| strongly endothermic | heated multitubular reactor, adiabatic stages with reheating | supplying heat limits the rate |
| equilibrium-limited | staged beds with inter-stage temperature control, or a membrane reactor removing a product | shifting equilibrium or temperature per stage |
| catalyst deactivates in minutes to hours | fluidised or moving bed with regeneration | a fixed bed would need constant swapping |
| gas-liquid reaction | by Hatta number: fast (reaction in the film) -> large interfacial area (packed column, bubble column); slow (in the bulk) -> large liquid hold-up (bubble column, stirred tank) | see `fidelity.md` |
| gas-liquid-solid catalytic | slurry reactor (fine catalyst, good heat removal) or trickle bed (fixed catalyst, near plug flow, wetting issues) | |
| hazardous or very fast chemistry, small scale | microreactor | heat and mass transfer by short distances |

These are tendencies, not rules. Say which consideration dominates for the
user's case, with a number where possible (adiabatic temperature rise, Hatta
number, deactivation time against residence time).

## Which model suits the reactor

| Reactor | Usual starting model | Climb to | Structure codes |
|---|---|---|---|
| batch, stirred tank | ideal mixing ODE / algebra | segregation or micromixing models; heat balance with jacket | S1 |
| tubular, homogeneous | plug flow | axial dispersion; radial profiles (laminar flow, Graetz) | S2, S4, S6 |
| packed bed | pseudo-homogeneous plug flow | + dispersion, film, particle, radial (the ladder in `fidelity.md`) | S2, S4, S7, S8, S6 |
| multitubular cooled bed | one tube, 1-D pseudo-homogeneous with wall heat transfer | 2-D radial; heterogeneous | S2, S6, S8 |
| fluidised bed | two-phase (bubble and emulsion) model | bubble-size and hydrodynamic correlations | S7 |
| bubble column, slurry | axial dispersion per phase with interphase transfer | gas hold-up and bubble-size closures | S4, S7 |
| trickle bed | plug flow with wetting efficiency and interphase transfer | partial wetting, liquid maldistribution | S7, S8 |
| membrane reactor | two coupled plug-flow channels with a flux law across the membrane | membrane as its own domain; pressure drop; heat | S7, S10 |
| monolith, catalytic converter | one channel, 1-D with film transfer to the washcoat | 2-D channel; washcoat diffusion | S6, S7, S8 |
| adsorber, chromatographic column | axial dispersion with linear driving force | pore diffusion, non-linear isotherms, sharp fronts | S4, S5, S8 |

Couplings between domains (membrane, washcoat, particle, two phases) are the
subject of the `model-patterns` skill once it exists; until then see
`assembly-styles.md` (block assembly) in the `conventions` skill.
