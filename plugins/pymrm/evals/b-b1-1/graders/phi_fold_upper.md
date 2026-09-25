---
type: llm
weight: 3
focus: { source: file, path: answers.json }
---

PASS if answers.json contains the key "phi_fold_upper" with a numeric value between 0.56607132 and 0.57750711 (inclusive).
FAIL otherwise, including when the key is missing or the value is not a number.
