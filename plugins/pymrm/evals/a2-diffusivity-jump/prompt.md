---
description: "P2: composite slab; arithmetic face mean converges at first order"
tags: [trap]
max_turns: 80
timeout_seconds: 1800
allowed_tools: [Read, Glob, Grep, Skill, Agent, TodoWrite]
---

A two-layer membrane: layer 1 occupies 0 < x < 0.5 mm with D1 = 1.0e-9 m2 s-1,
layer 2 occupies 0.5 mm < x < 1.0 mm with D2 = 1.0e-10 m2 s-1. The concentration
is 1.0 mol m-3 at x = 0 and 0 at x = 1.0 mm; concentration is continuous across
the interface (no partitioning). Compute the steady flux through the membrane
with a finite-volume model, accurate to 0.1 %, in nmol m-2 s-1.

Use pymrm (installed in the Python environment on PATH). Save the model as
model.py. End your final message with the line(s) shown, and nothing after them.
ANSWER: <flux in nmol m-2 s-1, 4 significant digits>
