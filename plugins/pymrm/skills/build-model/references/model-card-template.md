# Model card template (`model_card.md`)

The model card is what the user keeps. Lead with the answer. Every number in it
is printed by the code (copy it from the output, do not retype or round it by
hand) and every claim points to where it was checked.

```markdown
# Model card: <short name>
Date: <date>   pymrm version: <x.y.z>   Files: <model .py/.ipynb, spec.md>

## Answer
<one paragraph: the answer to the user's question, with the headline numbers
and their uncertainty>

## What was modelled
<one paragraph plus the structure codes; link spec.md for equations>
Included: <phenomena>
Excluded, and the estimated effect of each: <from scoping.md>

## Results
<table of outputs; figures produced by the notebook>

## How it was checked
| Assertion | Result | Verifier verdict | Evidence (command or cell) |
|---|---|---|---|
| A1 ... | ... | met / not met / insufficient evidence / blocked | ... |
Observed orders: <space, time>
Second independent route: <what, agreement>

## What rests on assumed values
| Parameter | Label | Headline at 0.5x | at 2x | Change needed to alter the conclusion |
|---|---|---|---|---|

## Validity limits
<operating range covered; where the correlations and kinetics stop being valid>

## What this model does not establish
<e.g. not validated against measurements; no dynamics; single steady-state
branch searched in 500-700 K only>

## Open items
<anything not met, blocked, or deferred; what would settle it>
```

Rules:
- A verdict of `not met`, `insufficient evidence` or `blocked` appears as such,
  in the table and in the answer if it touches the headline.
- "Validated" means compared with independent measurements. Agreement with a
  closed form or a second numerical route is "verified". Do not mix them up.
