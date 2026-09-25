---
type: llm
weight: 3
focus: { source: file, path: answers.json }
---

PASS if answers.json contains the key "p0_critical_at_625K" with a numeric value between 0.016428287 and 0.016593396 (inclusive).
FAIL otherwise, including when the key is missing or the value is not a number.
