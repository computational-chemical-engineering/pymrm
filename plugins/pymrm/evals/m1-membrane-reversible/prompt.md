---
description: "pattern: countercurrent membrane reactor, reversible reaction, one monolithic system"
tags: [pattern]
max_turns: 120
timeout_seconds: 3000
allowed_tools: [Read, Glob, Grep, Skill, Agent, TodoWrite]
---

A membrane reactor has two channels along 0 < z < 1 (dimensionless). In the
retentate, which flows in +z with unit velocity, the reversible reaction
A <-> B runs at rate Da (c_A - c_B / K) with Da = 2 and K = 2. Only B permeates
the membrane, with flux St (c_B - p_B) per unit retentate volume, St = 3. The
permeate carries only B and flows in -z with unit velocity (countercurrent);
the volume ratio of permeate to retentate is 1. Both channels have axial
dispersion with Peclet number 50. The retentate feed at z = 0 is c_A = 1,
c_B = 0; the permeate sweep enters at z = 1 free of B. Use Danckwerts
conditions at every inlet and zero gradient at every outlet.

Report the steady outlet concentrations, accurate to 0.3 %: retentate A and B
at z = 1, and permeate B at z = 0. Keys: ret_A_out, ret_B_out, perm_B_out.

Use pymrm. Save your code in the working directory and write the requested
numbers to answers.json in the working directory. I will not be available for
questions; state any assumption you make.
