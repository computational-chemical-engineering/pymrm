---
type: llm
focus: trace
---

PASS if the particle and bulk equations are solved as one coupled system (for
example a monolithic Newton system, or Newton with the particle unknowns
eliminated by a Schur complement), and the particle surface condition uses the
local bulk concentration. FAIL if the particle is replaced by an assumed
effectiveness factor without solving it, or the coupling is iterated outside
Newton without a reported converged coupling residual.
