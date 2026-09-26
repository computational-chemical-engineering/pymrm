---
type: llm
weight: 3
focus: { source: file, path: answers.json }
---

PASS if answers.json contains the key "ret_A_out" with a numeric value between 0.325424 and 0.328694 (inclusive).
FAIL otherwise, including when the key is missing or not a number.
