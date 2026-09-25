---
type: llm
weight: 3
focus: { source: file, path: answers.json }
---

PASS if answers.json contains the key "T_M_at_625K" with a numeric value between 656.51834 and 656.71834 (inclusive).
FAIL otherwise, including when the key is missing or the value is not a number.
