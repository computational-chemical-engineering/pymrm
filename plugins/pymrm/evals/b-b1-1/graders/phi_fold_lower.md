---
type: llm
weight: 3
focus: { source: file, path: answers.json }
---

PASS if answers.json contains the key "phi_fold_lower" with a numeric value between 0.29434793 and 0.30029435 (inclusive).
FAIL otherwise, including when the key is missing or the value is not a number.
