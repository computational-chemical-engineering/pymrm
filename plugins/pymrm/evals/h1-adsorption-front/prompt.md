---
description: "hard: Langmuir adsorption front; numerical dispersion inflates its width"
tags: [hard]
max_turns: 150
timeout_seconds: 3000
allowed_tools: [Read, Glob, Grep, Skill, Agent, TodoWrite]
---

A packed adsorption column, length 0.20 m, bed voidage 0.40, interstitial gas
velocity 0.010 m/s, axial dispersion coefficient 1.0e-5 m2/s, is initially
clean. At t = 0 the feed switches to 10 mol/m3 of an adsorbing species. The
isotherm is Langmuir, q* = q_m K c / (1 + K c) with q_m = 2000 mol per m3 of
solid and K = 0.5 m3/mol, and uptake follows a linear driving force,
dq/dt = k (q* - q) with k = 0.05 1/s. Isothermal, constant velocity, Danckwerts
boundary conditions.

Report, accurate to 1 %: the time at which the outlet concentration reaches
5 % of the feed (key t_5pct, s), and the width of the breakthrough front, the
time from 5 % to 50 % of the feed concentration at the outlet (key
front_width_s, s).

Use pymrm. Save your model code as model.py in the working directory and write
the requested numbers to answers.json there. I will not be available for
questions; state any assumption you make.
