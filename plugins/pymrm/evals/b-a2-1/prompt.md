---
description: "held-out A2.1"
tags: [heldout]
max_turns: 150
timeout_seconds: 3600
allowed_tools: [Read, Glob, Grep, Skill, Agent, TodoWrite]
---

An isothermal packed tubular reactor carries out an irreversible first-order reaction A -> products with no volume change. Model it as a steady 1-D axial-dispersion (dispersed plug flow) reactor: constant superficial velocity u, constant axial dispersion coefficient D and constant rate constant k throughout the bed. The feed pipe upstream and the exit pipe downstream have no dispersion and no reaction (a 'closed' vessel), and the feed concentration is c_in. In dimensionless form the bed has Peclet number Pe = uL/D = 2.667 and Damkohler number Da = kL/u = 2, with L the bed length. Use pymrm.

(a) What fraction of A leaves the bed unconverted, c(L)/c_in? (b) A colleague instead imposes the feed concentration directly at the bed entrance, c(0) = c_in, keeping everything else the same. By what relative amount does that version over-predict the exit concentration, i.e. what is c_colleague(L)/c_correct(L) - 1?

Write your final answers to answers.json in the working directory as a JSON object with exactly these keys (numbers, in the units asked): dirichlet_inlet_error_at_ww_fig4. Save the model code in the working directory as well. I will not be available to answer questions; make and state reasonable assumptions.
