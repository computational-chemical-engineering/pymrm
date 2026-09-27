---
type: llm
focus: { source: file, path: model.py }
---

PASS if every NumJac in the file is constructed with a shape that has a trailing
field axis (for example (n, 1)) or with an explicit stencil that gives a banded or
diagonal Jacobian, so the Jacobian is not dense.
FAIL if a NumJac is constructed with a bare one-dimensional shape such as (n,) or
(4000,), or with axes_diagonals on a one-dimensional shape, or if model.py is missing.
