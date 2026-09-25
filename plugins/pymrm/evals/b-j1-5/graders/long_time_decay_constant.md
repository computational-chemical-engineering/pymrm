---
type: llm
weight: 3
focus: { source: file, path: answers.json }
---

PASS if answers.json contains the key "long_time_decay_constant" with a numeric value between 9.8498765 and 9.8893549 (inclusive).
FAIL otherwise, including when the key is missing or the value is not a number.
