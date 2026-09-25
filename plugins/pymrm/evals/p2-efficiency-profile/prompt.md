---
description: "profile: efficiency; many solves, fast and still correct"
tags: [profile]
max_turns: 100
timeout_seconds: 2400
allowed_tools: [Read, Glob, Grep, Skill, Agent, TodoWrite]
---

I need the effectiveness factor for steady diffusion with second-order reaction in a catalyst slab of half-thickness
L = 1 mm (symmetry at the centre plane, surface concentration 5 mol m-3),
D_eff = 2e-9 m2 s-1, k2 = 4e-4 m3 mol-1 s-1. over a grid of 2000 combinations of
surface concentration (0.5 to 50 mol m-3) and k2 (1e-5 to 1e-2 m3 mol-1 s-1),
as fast as possible, accurate to 0.1 %. Use pymrm. Save the code as sweep.py and
the results as results.csv (columns c_s, k2, eta) in the working directory, and
report the wall time. I will not be available for questions.
