---
type: llm
focus: last_message
---

PASS only if ALL hold:
- It clearly disagrees with using a single CSTR for this purpose, early in the
  answer.
- The reason given is that runaway in a cooled tube is a hot-spot phenomenon
  along the axis (parametric sensitivity to coolant or inlet temperature and
  inlet concentration), which a well-mixed model cannot represent.
- It recommends at least a 1-D pseudo-homogeneous plug-flow model with wall
  cooling, with the critical coolant temperature located by root-finding or
  bisection on that model, and says when to climb further (radial gradients,
  catalyst particle).
- It says what would make the CSTR acceptable (for example a quick screening
  bound) rather than dismissing it without reason.
FAIL otherwise.
