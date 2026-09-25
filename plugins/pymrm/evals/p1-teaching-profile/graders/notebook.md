---
type: llm
focus: { source: file, path: model.ipynb }
---

PASS only if ALL hold:
- Markdown cells explain the model and each discretisation step before the code
  that implements it.
- The gradient, divergence and reaction Jacobian are separate, named matrices
  (built with pymrm operators, e.g. construct_grad, construct_div, NumJac on the
  reaction only) and the total Jacobian is visibly their sum.
- The boundary conditions are written as pymrm dictionaries with the physical
  equation beside each.
- A grid-refinement study with an observed order is shown, and at least one
  check against a limit or second route (e.g. first-order closed form, shooting).
FAIL if the Jacobian comes from one NumJac on the whole residual, if the
explanation is missing, or if the file is missing.
