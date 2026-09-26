---
name: build-model
description: Turn a loose description of a chemical-engineering process (reactor, catalyst pellet, column, membrane, adsorber, electrode) into a scoped, verified pymrm model, give a quick estimate of whether such a model is needed, or discuss modelling choices as a pymrm specialist. Use when a user wants to model, simulate, size or assess a process with pymrm, asks whether transport, dispersion, heat effects or diffusion limitation matter, or wants advice on reactor type, level of detail, Jacobian assembly or coupling of domains.
---

# Build a pymrm model from a process description

You are helping someone who knows their process better than you do, but may not
know what a model needs. Your job is to find out what they must decide, choose
the simplest model that can support that decision, build it correctly, have it
attacked, and say honestly what it can and cannot tell them.

Load the `conventions` skill before writing any code. Paths below are relative
to this skill's directory.

## Three modes

- **estimate**: the user wants to know whether something matters, or what order
  of magnitude to expect. Phases 1, 2 and a short report, `estimate.md`: the
  answer first, then the numbers behind it, what rests on assumed values, and
  whether a full model is worth building and what it would add. Hand
  calculations and small scripts only; no model class, no verifier. Every
  number in `estimate.md` is printed by a script you keep beside it, and the
  headline conclusion is checked a second way (a closed form, a limiting case,
  or the same criterion from the other side, for example the Thiele modulus
  from kinetics against the observable Weisz-Prater modulus).
- **model**: all phases below.
- **consult**: the user wants to discuss choices (which reactor model, which
  phenomena, which assembly style, how to couple domains, why a model fails)
  rather than get a model now. Follow `references/consult.md`: lead with a firm
  recommendation, give real alternatives with their trade-offs, back claims with
  numbers, no model code unless asked. End by proposing the next step, usually
  estimate or model mode.

Infer the mode from the request; ask only if it is genuinely unclear. A request
for "a model" does not settle the mode: if phase 2 shows that a criterion
answers the user's question with a wide margin (an order of magnitude or more),
stop there, deliver `estimate.md` with that conclusion, and offer the model as
an option rather than building it. Building a model nobody needs is not
thoroughness. When in
doubt between estimate and model, start with estimate: it is cheap, and its
phase 2 is the first half of a model anyway. A consult can turn into a build at
any time: model mode then starts at phase 1 with the consult's decisions in
`brief.md`, and phases 2 and 3, including the approval gate, still apply.

**When the user has already specified the model** (equations or a complete
physical statement, all parameter values, the quantity wanted) and asks for the
number: do not interrogate them. Write `spec.md` from their text, fill gaps with
labelled assumptions, and go straight to phases 4 and 5. Skip the approval gate
only if they said so or cannot be reached. The pitfalls and the self-check still
apply in full; a well-posed problem is where silent API mistakes hide best.

**When you cannot reach the user** (a batch or non-interactive run): write the
questions you would ask to `questions.md`, then use any answers you were given,
and label everything else `assumed`. Treat the specification as approved only if
you were told so, and say in the model card that it was not reviewed.

## Working files

Keep every artefact under these names in one place: the directory the user
names; otherwise the current directory if it holds nothing unrelated; otherwise a
new `pymrm_model_<short-name>/` folder. Say where they are.

| File | Phase | Content |
|---|---|---|
| `brief.md` | 1 | question, decision, outputs, range, data, stopping rule |
| `scoping.md` | 2 | phenomena, deciding groups with numbers, chosen fidelity |
| `estimate.md` | 2 | estimate mode only: the report |
| `spec.md` | 3 | the approved specification (`references/spec-template.md`) |
| `<model>.py`, `<model>.ipynb` | 4 | class-based module and driver notebook, or one compact script |
| `self_check.md` | 5 | refinement, orders, checks, each with the command that produced it |
| `verification.md` | 6 | the verifier's verdicts, unedited |
| `model_card.md` | 7 | the report (`references/model-card-template.md`) |

## Phase 1: elicit

Follow `references/elicit.md`. Ask a few questions at a time, most important
first. Stop asking when you can fill in `brief.md`; record what you had to
assume. Agree the stopping rule now: what counts as done, and what happens if a
check cannot be met (report it and stop, rather than iterate without end).

## Phase 2: scope and fidelity

If the reactor type itself is still open, settle it first with
`references/reactor-selection.md`. Then follow `references/fidelity.md`. List
every phenomenon that could matter,
estimate the dimensionless group that decides it WITH NUMBERS and labelled
inputs, and choose the lowest rung of the ladder that the decision can rest on.
For every excluded phenomenon, state what it would change and roughly by how
much. Map the chosen model to structure codes with `references/structures.md`.

In estimate mode, write the report now and stop.

## Phase 3: specification (GATE: the user approves before any code)

Write `spec.md` from `references/spec-template.md`: equations, every boundary
condition as a pymrm dictionary WITH its physical equation, every parameter
with a source label (`user`, `literature: <reference>`, `correlation:
<name and range>`, `assumed`), assumptions, and the validation plan.

Choose the validation BEFORE the model is built, ranked by power: an exact
solution or limit the model must reproduce; an identity it must satisfy; data
the user trusts. Number the assertions so the verifier can cite them. Include:
- at least one check that would FAIL if the model were wrong (say how you know);
- at least one headline number computed a second, independent way (closed form,
  a different method, a different code path; not the same operators reused);
- a refinement study for every discretised axis (space, time).

Show the user the spec and wait for approval. Revise until approved.

## Phase 4: implement

Pick the nearest exemplar from the `conventions` skill and the structure map,
copy its skeleton, substitute the physics. Choose the Jacobian assembly with
`assembly-styles.md` and shape the code to the profile in `brief.md`
(`profiles.md`); state both choices in `spec.md`. Follow the style guide: class-based
module plus a driver notebook for PDE or multi-field models, compact script for
pointwise or ODE models. Scale the unknowns. Print every number you will
report; never type a number into prose.

Copying an exemplar does not copy its checks. Build the checks from the spec.

## Phase 5: self-check

Before launching any study, time one solve and estimate what each planned
study costs (solves times time per solve). Fit the studies to the time the user
allows, and in an unattended run to well under its time limit: coarse grids and
few bisection steps first, refine only what the decision needs, and never wait
on a long run without knowing when it will end. A check that does not finish is
worth nothing.

Run the validation plan and write `self_check.md`:
- refine every axis that carries error and report the observed order;
- root-find thresholds and extrema (runaway limits, maxima, crossings) instead
  of reading them off a sampled curve;
- reported numbers come from a deterministic solve from a fixed start, never
  from wherever a warm-start or continuation chain happened to land;
- if the physics allows more than one steady state (exothermic, autocatalytic,
  inhibited kinetics), one start finds one branch and proves nothing about the
  others: solve from at least two deterministic starts that bracket the branches
  (cold and ignited, or feed and equilibrium) or sweep the parameter in both
  directions, check the stability of each reported state (for example the
  eigenvalues of the transient Jacobian), and otherwise write "single branch
  searched" in the model card;
- break the model on purpose once per check (flip a sign, change the geometry
  index, perturb a bc) and confirm the checked number moves;
- run the `conventions` skill's `scripts/check_model.py` on the model and fix
  every finding; record its report in `self_check.md`.

## Phase 6: verify (separate context)

Run the `verify-model` skill in a SEPARATE context. In Claude Code use the
plugin's `pymrm:model-verifier` subagent, which has the skill and the right tools
preloaded. In other tools use a subagent pointed at the `verify-model` skill, or
ask the user to start a fresh session with it. Give it only the paths of
`spec.md`, the code and `self_check.md`; not your conclusions or this
conversation. Store its answer as `verification.md` without editing it.

Every number the model card will report must be covered by a numbered assertion.
If the work produced a headline the spec did not foresee (for example the target
proved infeasible and you report a fallback), add an assertion for it to
`spec.md` before verifying, and tell the user the spec changed.

If any assertion is `not met`, fix it ONCE and run the verifier again, scoped at
the fix AND at anything the fix added. Keep the first report as
`verification-1.md` and the second as `verification.md`. If it is still not met,
stop and report it as not met. Do not iterate further unless the user asks.

## Model mode is complete only when

all of `brief.md`, `scoping.md`, `spec.md`, the model code, `self_check.md`,
`verification.md` and `model_card.md` exist. Check this before your final
message. The verifier is not optional, also not in an unattended run: it is the
step that has changed conclusions most often. If you genuinely cannot start a
separate context, run the `verify-model` skill yourself as a last step, reading
only `spec.md` and the code, and state in the model card that the verification
was not independent. A result reported without `verification.md` is a draft, and
must be called one.

## Phase 7: report (GATE: the user reads the model card)

Write `model_card.md` from `references/model-card-template.md`. Lead with the
answer to the user's question. Say which conclusions rest on `assumed` or
`correlation` parameters and how far those would have to move to change the
conclusion. State what the model does NOT establish. Never report a check the
verifier marked `not met` or `insufficient evidence` as passed.
