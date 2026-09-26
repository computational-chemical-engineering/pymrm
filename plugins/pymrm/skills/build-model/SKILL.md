---
name: build-model
description: Build a pymrm model from a description of a chemical-engineering process (reactor, catalyst pellet, column, membrane, adsorber, electrode), give a quick estimate of whether such a model is needed, discuss modelling choices as a pymrm specialist, or, on request, produce a documented and independently verified model. Use when a user wants to model, simulate, size or assess a process with pymrm, asks whether transport, dispersion, heat effects or diffusion limitation matter, or wants advice on reactor type, level of detail, Jacobian assembly or coupling of domains.
---

# Build a pymrm model from a process description

You are helping someone who knows their process better than you do. Find out
what they need to know, choose the simplest model that answers it, build it
correctly with the house conventions, check it, and say honestly what it can and
cannot tell them. Keep the process proportionate: most questions need a correct,
checked answer, not a document trail.

Load the `conventions` skill before writing any code. Paths below are relative
to this skill's directory.

## Four modes

- **direct** (the default): build and check the model, and answer. Follow the
  "Direct mode" section below.
- **estimate**: the user wants to know whether something matters, or what order
  of magnitude to expect. Scope with `references/fidelity.md`, compute the
  deciding groups with a small script, and answer: the conclusion first, the
  numbers behind it, what rests on assumed values, and whether a model would
  add anything. Check the headline a second way (a closed form, a limit, or the
  same criterion from the other side). If a criterion settles the question by a
  wide margin, stop there even if the user asked for "a model", and say why.
- **consult**: the user wants to discuss choices (which reactor model, which
  phenomena, which assembly style, how to couple domains, why a model fails).
  Follow `references/consult.md`: a firm recommendation first, real alternatives
  with their trade-offs, numbers behind claims, no model code unless asked.
- **full**: a documented, independently verified model, for decisions with real
  stakes (design sign-off, safety limits, publication) or when the user asks for
  a specification, verification or report. Follow the "Full mode" section. Do
  not start it unasked: when the stakes look high, offer it in one sentence and
  continue in direct mode unless the user accepts.

Infer the mode from the request; ask only if it is genuinely unclear.

## Direct mode

1. **Understand the question.** If the problem is fully specified (equations
   or a complete physical statement, parameter values, the quantity wanted),
   do not interrogate the user. If an input that changes the answer is missing,
   ask for it, a few questions at a time with a proposed default
   (`references/elicit.md` has the questions worth asking); otherwise assume a
   sensible value and label it. In a batch or non-interactive run, write the
   questions you would ask to `questions.md`, use any answers you were given,
   and label everything else `assumed`.
2. **Scope, briefly.** For a loosely posed problem, estimate the groups that
   decide which phenomena matter (`references/fidelity.md`: Thiele,
   Weisz-Prater, Mears, Peclet, Hatta, Biot) with the user's numbers, and
   include what matters. A phenomenon the user's intrinsic data leave out (for
   example diffusion inside large pellets) is exactly what this step catches.
   If the reactor type is open, see `references/reactor-selection.md`.
3. **Build.** Start from the nearest exemplar (`conventions`, `model-patterns`,
   `references/structures.md`), follow `assembly-styles.md` and the profile
   (`profiles.md`), and the style guide's format rule. Scale the unknowns. Print
   every number you report.
4. **Check** (the self-check below, proportionate to the question): the checker,
   `residual_check` on the reported solution, a refinement study of the reported
   numbers, one independent check (a limit, a closed form, a second method),
   and every stated condition mapped to the line of code that implements it.
5. **Answer.** Lead with the answer. Then, briefly: how it was checked (with the
   numbers), what rests on assumed values, and what the model does not
   establish. Save the model code; other files only if they help the user. Offer
   full mode if the result feeds a decision with real stakes.

## Self-check (all modes that build a model)

**Analysis before numerics.** Derive what can be derived: limits, asymptotes,
lag constants, thresholds of simple sub-models, conserved quantities. Use the
model to confirm them, not to discover them by fitting. If you must fit an
asymptote, fit the known functional form and show that the constant is stable
across fitting windows.

**Budget.** Time one solve before launching any study, estimate what each study
costs, and fit the studies to the time available (well under the limit in an
unattended run). Run them in the foreground with a `timeout`; do not start
background jobs and poll them with `sleep`. A check that does not finish is
worth nothing.

**Checks:**
- refine every axis that carries error in the reported numbers, with
  `observed_orders` (`numerics.md`), and report the observed order;
- root-find thresholds and extrema (`find_roots`) instead of reading them off a
  sampled curve;
- reported numbers come from a deterministic solve from a fixed start, never
  from wherever a warm-start or continuation chain happened to land;
- if the physics allows more than one steady state, solve from starts that
  bracket the branches, check the stability of each reported state, or say
  "single branch searched";
- `check_jacobian` once at a state away from the solution when you wrote the
  Jacobian yourself;
- run the `conventions` skill's `scripts/check_model.py` and fix every finding;
- for every notebook you deliver, run `scripts/check_notebook_math.py` and fix
  every error;
- list every condition the user stated (geometry, assumptions such as constant
  density, boundary conditions, units, definitions of the reported quantities)
  and point to the code line that implements each: a second numerical route
  shares your reading of the problem and cannot catch a misreading;
- break the model once on purpose for the check you rely on most, and confirm
  the checked number moves.

## Full mode

For decision-grade work, add the following to the steps above, keeping every
artefact in one place (the directory the user names; otherwise the current
directory if it holds nothing unrelated; otherwise `pymrm_model_<short-name>/`):

| File | Content |
|---|---|
| `brief.md` | question, decision, outputs, range, data, stopping rule (`references/elicit.md`) |
| `scoping.md` | phenomena, deciding groups with numbers, chosen fidelity (`references/fidelity.md`) |
| `spec.md` | the specification (`references/spec-template.md`), APPROVED by the user before any code |
| model code | by the style guide's format rule |
| `self_check.md` | the self-check above, each result with the command that produced it |
| `verification.md` | the independent verifier's verdicts, unedited |
| `model_card.md` | the report (`references/model-card-template.md`) |

- **Specification first.** Equations, every boundary condition as a pymrm
  dictionary with its physical equation, every parameter with a source label
  (`user`, `literature: <ref>`, `correlation: <name and range>`, `assumed`), and
  a validation plan chosen before the model is built: numbered assertions,
  including at least one check that would fail if the model were wrong and one
  headline computed a second, independent way. Wait for approval (in a
  non-interactive run, treat it as approved only if told so, and say it was not
  reviewed).
- **Independent verification.** Run the `verify-model` skill in a separate
  context: in Claude Code the plugin's `pymrm:model-verifier` subagent; in other
  tools a subagent pointed at `verify-model`, or a fresh session. Give it only
  `spec.md`, the code and `self_check.md`. Every number the model card reports
  must be covered by an assertion; add assertions for headlines the spec did
  not foresee. If an assertion is `not met`, fix it once and verify again
  (keep the first report as `verification-1.md`); if it is still not met,
  report it as not met.
- **Model card.** Lead with the answer; say which conclusions rest on assumed
  or correlated parameters and how far they would have to move to change the
  conclusion; state what the model does not establish; never report a check the
  verifier did not accept as passed. Full mode is complete only when all files
  above exist; without `verification.md` the result is a draft and must be
  called one.
