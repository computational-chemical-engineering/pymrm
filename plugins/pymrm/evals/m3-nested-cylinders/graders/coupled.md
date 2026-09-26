---
type: llm
focus: { source: file, path: model.py }
---

PASS if the particle equation is actually solved (not replaced by an assumed or
closed-form effectiveness factor) and its surface condition uses the local bulk
concentration, by any exact coupling: one monolithic Newton system, Newton with
the particle eliminated by a Schur complement, or the particle solved inside the
bulk equation's right-hand side at every evaluation. FAIL if the particle is not
solved, or if the coupling is iterated to a tolerance without reporting that it
converged, or if model.py is missing.
