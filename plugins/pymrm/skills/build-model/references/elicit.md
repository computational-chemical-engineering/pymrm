# Eliciting what the model must do

Users describe their process, not their model. Turn the description into a brief
by asking about the purpose first and the physics last. Ask two or three
questions at a time. Offer a default with each question, so a user who does not
know can say "yes, use that".

## The questions, in order of importance

1. **What decision hangs on the answer?** (choose a catalyst size, check a
   reactor will not run away, size a membrane area, interpret lab data, teach a
   concept). This decides everything else.
2. **Which outputs, and how accurately?** Conversion, selectivity, outlet
   temperature, hot-spot temperature, pressure drop, breakthrough time,
   effectiveness factor. A 20 % answer and a 1 % answer need different models.
3. **What operating range?** Temperatures, pressures, flows, compositions,
   sizes. The model is only claimed valid inside this range.
4. **Steady state or dynamics?** Start-up, cycling, breakthrough, deactivation
   need time; design at nominal conditions usually does not.
5. **What does the user know?** Geometry, kinetics (rate law, parameters, where
   they came from), transport properties, measured data to compare against.
   Ask for units every time.
6. **What is out of scope?** Things the user already knows are negligible, or
   explicitly does not want.
7. **How much time and effort?** Minutes (estimate mode) or a full model with
   verification.
8. **What should the code be good at?** The profile: default, efficiency (many
   solves), flexibility (will grow), readability (others must trust it) or
   teaching (explains the method). See the `conventions` skill's `profiles.md`;
   a project may set its own default.

## When to stop asking

Stop when `brief.md` can be filled in. Anything still unknown becomes an
assumption, written down with the value you chose and why. Do not stall on a
parameter: an `assumed` value with a sensitivity check is better than no model.

## Brief format (`brief.md`)

```markdown
# Brief: <short name>
Mode: estimate | model
Decision: <what the user will decide with this>
Outputs: <quantity, unit, required accuracy> (one line each)
Operating range: <variable: min to max, unit>
Dynamics: steady | transient (<which event>)
Known data: <what, source, units>
Out of scope: <list>
Assumptions made during elicitation: <value, reason>
Profile: default | efficiency | flexibility | readability | teaching (<source: user, project, default>)
Stopping rule: <what counts as done; what happens if a check cannot be met>
Questions asked: <list, with the user's answers>
```

## Things users often do not say, but you should ask

- Is the reaction exo- or endothermic, and how strongly (adiabatic temperature
  rise)? Heat effects change the model class.
- Particle size and tube diameter: they decide dispersion, wall effects and
  internal diffusion.
- Is there more than one phase (gas-liquid, gas-solid-liquid)?
- Are the kinetics intrinsic, or measured on the same particles (then they
  already contain internal diffusion)?
- Could there be more than one steady state (exothermic, autocatalytic,
  substrate-inhibited)? Then every branch must be found.
