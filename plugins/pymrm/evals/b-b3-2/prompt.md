---
description: "held-out B3.2"
tags: [heldout]
max_turns: 150
timeout_seconds: 3600
allowed_tools: [Read, Glob, Grep, Skill, Agent, TodoWrite]
---

A porous solid (semi-infinite slab, planar exposed face at y = 0) is made of dense spherical grains of radius R_g = 5.0e-3 cm with porosity (void fraction) S_g = 0.25. A reactant gas diffuses into the slab through the interstices with effective diffusivity D' = S_g D_p / F_T, where D_p = 8.0 cm2/s and tortuosity factor F_T = 2.75. Each grain reacts as a shrinking unreacted core of radius R'(y,t) <= R_g: an irreversible reaction first order in the local gas concentration C, with rate constant k per unit core surface, in series with diffusion through the grain's product shell with diffusivity D = 2.0e-4 cm2/s. The gas phase is quasi-steady (pore gas holdup negligible against the solid, molar density rho), isothermal, no film resistance: C = C_0 at y = 0 and C -> 0 deep inside. The extent of reaction is the equivalent penetration EP(t) = integral over y of [1 - (R'/R_g)^3] dy. Use pymrm; time can be reported as theta = k C_0 t / (rho R_g).

(1) For k = 20.0 cm/s: at long times the reaction occupies a zone that travels into the solid with a fixed thickness. What is that thickness in cm, measured between the depth where the local solid conversion 1 - (R'/R_g)^3 is 95 % and the depth where it is 5 %?

(2) For k = 0.005 cm/s (other values unchanged): EP grows linearly in t at early times, and at late times approaches a square-root law EP_inf(t) = sqrt(A (t - t_lag)) with its exact asymptotic constants A and t_lag. Give, in theta, (a) the time at which EP has fallen to 95 % of its initial linear growth (EP = 0.95 x initial slope x t), and (b) the time at which the late-time asymptote reaches 95 % of EP (EP_inf/EP = 0.95), beyond which it stays within 5 %.

Write your final answers to answers.json in the working directory as a JSON object with exactly these keys (numbers, in the units asked): zone_width_cm_fig7base, theta_early5_g05, theta_late5_g05. Save the model code in the working directory as well. I will not be available to answer questions; make and state reasonable assumptions.
