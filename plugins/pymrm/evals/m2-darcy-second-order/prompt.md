---
description: "pattern: pressure-velocity coupling with second-order kinetics"
tags: [pattern]
max_turns: 120
timeout_seconds: 3000
allowed_tools: [Read, Glob, Grep, Skill, Agent, TodoWrite]
---

Isothermal gas flows through a packed tube, 0 < z < 1 (dimensionless). The
superficial velocity follows Darcy's law, u = -dp/dz, and the gas is ideal with
total concentration c_t = p. The pressure is 1 at the inlet and 0.5 at the
outlet. Species A (inlet mole fraction 1) reacts irreversibly with rate
k c_A^2, k = 1, without change in the number of moles; axial dispersion is
negligible. Compute the pressure profile and the conversion of A together,
and report the conversion accurate to 0.2 %. Key: conversion.

Use pymrm. Save your model code as model.py in the working directory and write the requested
numbers to answers.json in the working directory. I will not be available for
questions; state any assumption you make.
