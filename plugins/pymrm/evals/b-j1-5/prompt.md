---
description: "held-out J1.5"
tags: [heldout]
max_turns: 150
timeout_seconds: 3600
allowed_tools: [Read, Glob, Grep, Skill, Agent, TodoWrite]
---

Adsorbent particles are spheres of radius r in which the adsorbed species moves by Fickian diffusion with constant diffusivity D; the isotherm is linear. At t = 0 a particle with zero loading is exposed to a step in fluid concentration, so its surface loading jumps to q* and stays there (no external film resistance, fluid not depleted). Let qbar(t) be the volume-averaged loading and tau = D t / r^2. A column model would replace the intraparticle diffusion by the linear driving force dqbar/dt = k (q* - qbar) with k = 15 D/r^2. Use pymrm for the diffusion problem.

Questions: (1) Over all times, what is the largest absolute difference in fractional uptake qbar/q* between the linear-driving-force prediction and the true diffusion solution? (2) At long times 1 - qbar/q* decays exponentially; what is its decay constant in units of D/r^2 (i.e. the rate constant an LDF model would need to match the tail)?

Write your final answers to answers.json in the working directory as a JSON object with exactly these keys (numbers, in the units asked): ldf_worst_abs_err, long_time_decay_constant. Save the model code in the working directory as well. I will not be available to answer questions; make and state reasonable assumptions.
