---
type: llm
weight: 3
focus: { source: file, path: answers.json }
---

PASS if answers.json contains the key "theta_late5_g05" with a numeric value between 0.48283288 and 0.50254034 (inclusive).
FAIL otherwise, including when the key is missing or the value is not a number.
