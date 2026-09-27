---
type: llm
focus: { source: file, path: model.py }
---

PASS if the two channels were solved as one coupled system (one Newton or one
linear solve containing both channels' unknowns), not by iterating between
separate channel solves. FAIL if the model iterates between channels, or if model.py is missing.
