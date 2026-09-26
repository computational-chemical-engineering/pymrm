---
type: llm
weight: 3
focus: { source: file, path: answers.json }
---

PASS if answers.json contains the key "conversion" with a numeric value between 0.62375 and 0.62625 (inclusive).
FAIL otherwise, including when the key is missing or not a number.
