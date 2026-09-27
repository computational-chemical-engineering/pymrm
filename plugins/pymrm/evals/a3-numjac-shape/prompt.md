---
description: "P3: single-field NumJac shape; (n,) is dense"
tags: [trap]
max_turns: 80
timeout_seconds: 1800
allowed_tools: [Read, Glob, Grep, Skill, Agent, TodoWrite]
---

A washcoat layer of half-thickness 0.5 mm (symmetric, no flux at the centre plane
x = 0) has reactant concentration 10 mol m-3 at its surface x = 0.5 mm. The
reactant diffuses with D_eff = 1.0e-8 m2 s-1 and reacts with second-order kinetics
r = k2 c^2, k2 = 0.08 m3 mol-1 s-1. Use a finite-volume grid of 4000 cells and
pymrm's NumJac for the reaction Jacobian. Report the effectiveness factor and the
wall-clock time of the solve.

Use pymrm (installed in the Python environment on PATH). Save the model as
model.py. End your final message with the line(s) shown, and nothing after them.
ANSWER: <effectiveness factor, 4 significant digits>
