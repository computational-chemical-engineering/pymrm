---
description: "hard: ternary Maxwell-Stefan film; Fick gets the N2 flux sign wrong"
tags: [hard]
max_turns: 150
timeout_seconds: 3000
allowed_tools: [Read, Glob, Grep, Skill, Agent, TodoWrite]
---

A capillary of length 0.086 m connects two large bulbs of a H2 / N2 / CO2 gas
mixture at 1 bar and 308.35 K (ideal gas). The composition at the capillary end
in bulb 1 (z = 0) is y_H2 = 0.0, y_N2 = 0.50086, y_CO2 = 0.49914; at the end in
bulb 2 (z = L) it is y_H2 = 0.49890, y_N2 = 0.50110, y_CO2 = 0.0. The binary
diffusivities are D(H2-N2) = 83.3e-6, D(H2-CO2) = 68.0e-6 and
D(N2-CO2) = 16.8e-6 m2/s. The total molar flux is zero (the bulbs are closed and
at equal pressure). Treat the capillary as steady and one-dimensional.

Report the steady molar fluxes of H2 and N2 in mol m-2 s-1, positive in the
direction from bulb 1 to bulb 2, accurate to 1 %. Keys: N_H2, N_N2.

Use pymrm. Save your model code as model.py in the working directory and write
the requested numbers to answers.json there. I will not be available for
questions; state any assumption you make.
