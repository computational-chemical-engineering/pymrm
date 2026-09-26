---
type: llm
weight: 3
focus: { source: file, path: answers.json }
---

PASS if answers.json contains the key "ret_B_out" with a numeric value between 0.338054 and 0.341452 (inclusive).
FAIL otherwise, including when the key is missing or not a number.
