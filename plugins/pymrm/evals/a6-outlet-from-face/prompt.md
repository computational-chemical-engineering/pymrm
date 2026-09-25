---
description: "P8: outlet read from the last cell; balance does not close"
tags: [trap]
max_turns: 80
timeout_seconds: 1800
allowed_tools: [Read, Glob, Grep, Skill, Agent, TodoWrite]
---

A liquid-phase tubular reactor, L = 0.5 m, superficial velocity 0.01 m s-1, axial
dispersion coefficient 1.0e-3 m2 s-1, first-order reaction k = 0.04 s-1, feed
concentration 2.0 mol m-3, Danckwerts boundary conditions. Use a finite-volume
model with 50 cells. Report the steady outlet conversion (4 decimals) and show
that the overall steady mass balance (in, out, consumed) closes; report its
relative residual.

Use pymrm (installed in the Python environment on PATH). Save the model as
model.py. End your final message with the line(s) shown, and nothing after them.
ANSWER_X: <conversion>
ANSWER_BALANCE: <relative residual>
