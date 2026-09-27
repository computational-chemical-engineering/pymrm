---
type: llm
focus: last_message
---

PASS only if ALL hold:
- The first sentence or two give a clear recommendation (not a neutral survey).
- It recognises that the Maxwell-Stefan flux is nonlinear in the state, so a
  NumJac on the flux part (full-residual or hybrid with the linear transport kept
  analytic) is appropriate there, while the channels and linear terms can stay
  operator-based or block-assembled.
- It warns that the NumJac stencil must cover neighbour and species coupling
  (axes_diagonals on a shape with at least two axes, axes_blocks over species),
  or equivalent.
- It gives at most two or three real alternatives with a one-line trade-off.
- It does not write a full model.
FAIL otherwise.
