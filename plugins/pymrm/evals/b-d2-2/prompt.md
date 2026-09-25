---
description: "held-out D2.2"
tags: [heldout]
max_turns: 150
timeout_seconds: 3600
allowed_tools: [Read, Glob, Grep, Skill, Agent, TodoWrite]
---

A multitubular fixed-bed reactor with wall cooling carries out a strongly exothermic gas-phase oxidation A + B -> products. Model one tube as a steady 1-D pseudo-homogeneous plug-flow reactor (no axial dispersion, no radial gradients, constant density and velocity). The rate per kg catalyst is r = exp(b - a/T) p_A p_B [kmol/(kg cat h)], with p in atm, a = 13636 K, b = 19.837; B is in large excess at constant p_B = 0.208 atm. Data: total pressure P = 1 atm, mean molar mass M = 29.48 kg/kmol, gas density rho_g = 1.293 kg/m3, catalyst bulk density rho_b = 1300 kg/m3, heat of reaction (-dH) = 307000 kcal/kmol, gas heat capacity c_p = 0.323 kcal/(m3 degC), superficial velocity u = 3600 m/h, overall wall heat-transfer coefficient U = 82.7 kcal/(m2 h degC), tube radius R = 0.0125 m. Coolant (wall) temperature T_w = 625 K, and the feed enters at T_0 = T_w. The tube is long enough that the hot spot always lies inside it. Use pymrm.

In the (T, p_A) plane every reactor is one trajectory, and the locus of hot-spot points (where dT/dz = 0) is a curve p_A(T) that is independent of the inlet. Take as the runaway criterion that the critical reactor is the one whose trajectory passes through the maximum of that locus. (1) What is the critical hot-spot temperature T_M? (2) What inlet partial pressure of A, p_A0, is critical at T_w = 625 K?

Write your final answers to answers.json in the working directory as a JSON object with exactly these keys (numbers, in the units asked): T_M_at_625K, p0_critical_at_625K. Save the model code in the working directory as well. I will not be available to answer questions; make and state reasonable assumptions.
