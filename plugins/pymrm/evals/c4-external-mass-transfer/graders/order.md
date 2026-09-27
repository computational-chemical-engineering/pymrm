---
arm: with-only
type: tool_order
before: { tool: Write, input_match: 'questions\.md' }
after: { tool: Read, input_match: 'persona\.md' }
---
