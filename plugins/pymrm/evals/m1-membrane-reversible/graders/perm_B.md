---
type: llm
weight: 3
focus: { source: file, path: answers.json }
---

PASS if answers.json contains the key "perm_B_out" with a numeric value between 0.331523 and 0.334855 (inclusive).
FAIL otherwise, including when the key is missing or not a number.
