---
type: llm
focus: { source: file, path: estimate.md }
---

PASS only if ALL hold:
- It leads with the answer to the user's question.
- It states which conclusions rest on assumed values and what is not established.
5. The run evaluates an external mass-transfer criterion (Mears or an equivalent
   film-drop estimate) with numbers, with the mass-transfer coefficient from a
   labelled correlation (a Sherwood number of at least 2 is the conservative
   bound), finds it passed by a large margin (orders of magnitude), and also
   checks internal diffusion.
6. It concludes that no PDE model is needed, explains why, and does not build a
   full reactor model.
FAIL if any point fails or the file is missing.
