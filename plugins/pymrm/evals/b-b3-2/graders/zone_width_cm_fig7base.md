---
type: llm
weight: 3
focus: { source: file, path: answers.json }
---

PASS if answers.json contains the key "zone_width_cm_fig7base" with a numeric value between 0.29212088 and 0.29802231 (inclusive).
FAIL otherwise, including when the key is missing or the value is not a number.
