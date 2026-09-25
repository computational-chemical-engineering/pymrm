---
type: llm
weight: 3
focus: { source: file, path: answers.json }
---

PASS if answers.json contains the key "dirichlet_inlet_error_at_ww_fig4" with a numeric value between 0.49425765 and 0.49922506 (inclusive).
FAIL otherwise, including when the key is missing or the value is not a number.
