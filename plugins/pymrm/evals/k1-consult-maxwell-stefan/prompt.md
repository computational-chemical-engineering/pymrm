---
description: "consult: Jacobian style for a Maxwell-Stefan membrane"
tags: [consult]
max_turns: 30
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, Agent, TodoWrite]
---

I am modelling a zeolite membrane with Maxwell-Stefan diffusion of a ternary
mixture across the membrane, coupled to feed and permeate channels. Should I
wrap my whole residual in NumJac, or build the Jacobians analytically from pymrm
operators like I usually do? I just want your advice, no code yet.
