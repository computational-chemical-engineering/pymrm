---
arm: with-only
type: llm
focus: last_message
---

PASS only if ALL hold:
- It leads with the answer to the user's question.
- It states which conclusions rest on assumed values and what is not established.
5. The run discovers that 90 % recovery is NOT reachable with a 1 bar hydrogen
   permeate at 10 bar feed (the hydrogen partial-pressure driving force vanishes
   at about 88.9 % recovery), says so clearly, and gives the user options
   (lower permeate pressure or sweep, higher feed pressure, a lower target).
6. For a reachable recovery it reports an area consistent with Sieverts' law in
   plug flow: about 0.81 m2 for 80 % and about 1.0 m2 for 85 % (within 5 %), or
   an area for another stated recovery that is consistent with these.
FAIL if any point fails or the file is missing.
