---
name: verify-model
description: Adversarially verify a finished pymrm model against its approved specification - rerun it, attack its baseline, check boundary conditions, refinement and independent routes, and return a verdict per numbered assertion (met, not met, insufficient evidence, blocked). Use in a fresh context after a pymrm model is built, or when asked to review or audit a pymrm model's results.
---

# Verify a pymrm model

You did not build this model and you do not trust it. Your job is to find what
is wrong with it before the user relies on it. In the pymrm gallery, every one of
six models needed fixes after this step, and on four the conclusion changed.

Load the `conventions` skill first; its `pitfalls.md` is part of your checklist.

## Inputs

- `spec.md`: the approved specification with numbered assertions. This is what
  you judge against.
- the model code (`.py` module, notebook or script);
- `self_check.md`: what the builder says it ran.

Do NOT read the builder's conclusions, the model card, or the conversation that
produced the model. If you were given them, ignore them. If `spec.md` or the code
is missing, your verdict is `blocked`: say what is missing and stop.

## Method

1. **Rerun everything yourself.** Execute the code from a clean state. Every
   number you cite comes from your own run, with the command that produced it.
   A number in `self_check.md` that you cannot reproduce is a finding.
2. **Check the spec against the physics, then the code against the spec.**
   For every boundary condition: read the physical equation, derive `a`, `b`,
   `d` yourself with the outward-normal rule, compare with the code. For every
   parameter: value, unit, label, and whether it is used where the spec says.
3. **Attack the baseline, not only the inputs.** A break row (perturb an input,
   watch a number move) shows sensitivity, never correctness. For every headline,
   ask: if this number were wrong at baseline, which check would reveal it? If
   none, compute it yourself by a route that shares no code with the model
   (closed form, limit, different method, `scipy` directly).
4. **Refine every axis that carries error**: space and time, separately and
   together. Report observed orders. If the reported value is not converged to
   the accuracy the spec asks for, the assertion is `not met`.
5. **Hunt the defect classes by name** (`references/attack-list.md`),
   including a misread problem statement: map every stated condition to code. Each one
   has been found repeatedly in real models.
6. **Check every check.** For each assertion the builder claims to meet, break
   the model on purpose in a way that assertion should detect and confirm the
   checked number moves. A check that does not move is decoration: say so.
7. **Read claims against evidence.** For each claim in the code comments,
   printed output and `self_check.md`: what statistic, over what range, with what
   held fixed? Does the evidence support the sentence, or only a weaker one?

## Budget

Verification must finish. Time one solve of the model first. Then spend the
time in this order: reproduce the headline numbers; run the cheapest check that
could overturn each headline (a limit, an independent route on a coarse grid, a
break row); only then re-run long studies, and only those whose result the
cheaper evidence cannot decide. Do not re-run a long study just because the
builder ran it; its log plus a spot check at one point is often enough. List in
the report what you did not re-run and why.

## Verdict per assertion

One line per numbered assertion in `spec.md`, plus one per extra finding:

- `met`: you reproduced it, and you confirmed that the check can fail.
- `not met`: it fails, or the check cannot fail, or the value is not converged.
- `insufficient evidence`: the evidence cannot decide it; name the smallest
  additional computation that would.
- `blocked`: an input is missing or the code does not run; say what.

Never invent a verdict to avoid an inconclusive one.

## Output format (`verification.md`)

```markdown
# Verification: <model name>
Verifier run: <date>, pymrm <version>, command(s): <...>

## Verdicts
| Assertion | Verdict | Evidence (your number, your command) | Can the check fail? (how you tested it) |
|---|---|---|---|

## Findings outside the assertions
<numbered; each: what, where (file:line), measured effect on a headline, fix>

## What this verification does not establish
<e.g. physics choices were not reviewed against data; only 500-600 K tested>
```

Keep it factual and short. Do not fix the model yourself; describe the fix.
