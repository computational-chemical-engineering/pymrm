---
type: llm
weight: 3
focus: { source: file, path: answers.json }
---

PASS if answers.json contains the key "c_out" with a numeric value between 0.390519 and 0.394443 (inclusive).
FAIL otherwise, including when the key is missing or not a number.
