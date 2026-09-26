---
description: "pattern: packed bed with resolved cylindrical particles and film"
tags: [pattern]
max_turns: 120
timeout_seconds: 3000
allowed_tools: [Read, Glob, Grep, Skill, Agent, TodoWrite]
---

A packed bed is modelled in plug flow along 0 < z < 1 (dimensionless):
dc_b/dz = -alpha * 2 * dc_p/dr at r = 1, with alpha = 0.5 and c_b(0) = 1.
At every position the catalyst is a long cylinder (radius 1, dimensionless)
with diffusion and reaction inside:
(1/r) d/dr (r dc_p/dr) = phi^2 c_p^1.5, phi = 2, symmetric at r = 0, and a
film at the surface: dc_p/dr = Bi (c_b - c_p) at r = 1 with Bi = 10.
Report the bulk concentration at the bed outlet, accurate to 0.3 %. Key: c_out.

Use pymrm. Save your code in the working directory and write the requested
numbers to answers.json in the working directory. I will not be available for
questions; state any assumption you make.
