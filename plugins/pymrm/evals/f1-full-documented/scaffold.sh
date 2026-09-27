#!/bin/bash
set -e
cat > persona.md <<'EOF'
Answers from the process engineer:
- Decision: membrane area to buy for a hydrogen recovery unit.
- Feed: 1.0 mol s-1, 50 mol % H2 in N2, 10 bar, 673 K.
- Permeate: pure hydrogen at 1.0 bar (no sweep gas, no vacuum pump planned).
- Membrane: Pd-Ag, Sieverts' law, hydrogen permeance 2.0e-3 mol m-2 s-1 Pa-0.5
  at 673 K (from the supplier's datasheet); nitrogen does not permeate.
- Tubular module, feed on the shell side, assume plug flow. Isothermal.
- Pressure drop and concentration polarisation: we do not know, probably small.
- We would like 90 % hydrogen recovery.
EOF
