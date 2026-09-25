---
type: llm
weight: 3
focus: { source: file, path: answers.json }
---

PASS if answers.json contains the key "eta_ignited" with a numeric value between 44.227958 and 44.67246 (inclusive).
FAIL otherwise, including when the key is missing or the value is not a number.
