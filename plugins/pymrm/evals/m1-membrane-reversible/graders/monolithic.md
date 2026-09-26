---
type: llm
focus: trace
---

PASS if the two channels were solved as one coupled system (one Newton or one
linear solve containing both channels' unknowns), not by iterating between
separate channel solves. FAIL if the model iterates between channels, or if the
transcript does not show how the channels were coupled.
