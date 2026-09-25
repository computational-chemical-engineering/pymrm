---
type: llm
weight: 3
focus: { source: file, path: answers.json }
---

PASS if answers.json contains the key "theta_early5_g05" with a numeric value between 0.12556189 and 0.1280985 (inclusive).
FAIL otherwise, including when the key is missing or the value is not a number.
