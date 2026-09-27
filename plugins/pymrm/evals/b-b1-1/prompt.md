---
description: "held-out B1.1"
tags: [heldout]
max_turns: 150
timeout_seconds: 3600
allowed_tools: [Read, Glob, Grep, Skill, Agent, TodoWrite]
---

A spherical porous catalyst pellet of radius R runs an exothermic, irreversible first-order reaction at steady state. The surface is held at the bulk gas concentration c_s and temperature T_s (no external film resistances). Inside, the reactant diffuses with constant effective diffusivity D_eff and heat is conducted with constant effective conductivity lambda_eff; the rate constant is Arrhenius, k(T) = k_s exp[-E/R_g (1/T - 1/T_s)]. Use the dimensionless groups reaction-diffusion modulus phi = R sqrt(k_s/D_eff), Prater number beta = (-dH) D_eff c_s/(lambda_eff T_s) = 0.6 and Arrhenius number gamma = E/(R_g T_s) = 20. The effectiveness factor eta is the actual pellet rate divided by the rate the whole pellet would have at surface concentration and temperature. Use pymrm.

Questions: (1) Over what range of phi does the pellet admit more than one steady state? Give the lower and upper limit of that phi-range. (2) At phi = 0.41232, what is the effectiveness factor of the steady state with the highest conversion (the hottest one)?

Write your final answers to answers.json in the working directory as a JSON object with exactly these keys (numbers, in the units asked): phi_fold_lower, phi_fold_upper, eta_ignited. Save the model code in the working directory as well. I will not be available to answer questions; make and state reasonable assumptions.
