---
type: llm
focus: { source: file, path: sweep.py }
---

PASS only if ALL hold:
- Constant operators are built once, outside the sweep loop.
- The Jacobian is not assembled by re-placing blocks or rebuilding operators per
  Newton iteration, and NumJac (if used) acts on local terms only.
- The grid resolution for 0.1 % accuracy was established by a refinement study
  (in the code or its printed output), not assumed.
- If solutions are warm-started across the sweep, some points are re-solved from
  a cold start and compared, or uniqueness is argued.
FAIL otherwise or if the file is missing.
