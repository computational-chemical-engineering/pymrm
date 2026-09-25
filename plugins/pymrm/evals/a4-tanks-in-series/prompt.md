---
description: "P6: stirred-tank outlet; zero-gradient bc is 0.4 % off"
tags: [trap]
max_turns: 80
timeout_seconds: 1800
allowed_tools: [Read, Glob, Grep, Skill, Agent, TodoWrite]
---

Six equal, ideally stirred tanks in series; total residence time 10 min; a
first-order reaction with k = 0.3 min-1. At t = 0 the feed switches from pure
solvent (all tanks at zero) to reactant at 1 mol L-1. Build a pymrm
finite-volume model with one cell per tank, so it can later be extended with
backmixing between tanks. Report the steady outlet conversion (4 decimals) and
the time at which the outlet concentration reaches half its steady value (min,
2 decimals).

Use pymrm (installed in the Python environment on PATH). Save the model as
model.py. End your final message with the line(s) shown, and nothing after them.
ANSWER_X: <conversion>
ANSWER_T50: <minutes>
