---
description: "P1: flux bc at the right end; wrong sign mirrors the profile"
tags: [trap]
max_turns: 80
timeout_seconds: 1800
allowed_tools: [Read, Glob, Grep, Skill, Agent, TodoWrite]
---

A catalyst coating of thickness L = 1 mm sits on an inert wall at x = 0, where a
downstream process keeps the reactant concentration at zero. At the free surface
x = L a reactant flux of 1.0e-4 mol m-2 s-1 enters the coating from the gas. The
reactant diffuses with D = 1.0e-9 m2 s-1 and is consumed by a first-order
reaction, k = 1.0e-3 s-1. What is the steady reactant concentration at the free
surface, in mol m-3, to 4 significant digits?

Use pymrm (installed in the Python environment on PATH). Save the model as
model.py. End your final message with the line(s) shown, and nothing after them.
ANSWER: <concentration in mol m-3, fixed-point, 2 decimals>
