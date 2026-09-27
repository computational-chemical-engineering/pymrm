---
type: llm
arm: with-only
focus: { source: file, path: model.py }
---

PASS if the bulk and particle unknowns are solved together in one Newton system
(monolithic, or with the particle eliminated by a Schur complement inside the
linear solve). FAIL if the bulk equation is integrated separately with the
particle solved inside its right-hand side, or if model.py is missing.
