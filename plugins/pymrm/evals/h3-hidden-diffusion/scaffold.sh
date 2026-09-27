#!/bin/bash
set -e
cat > persona.md <<'EOF'
Answers from the process engineer:
- Decision: whether the current design reaches 95 % conversion, or needs changes.
- Reactor: packed bed, 1.0 m long, bed voidage 0.40, superficial gas velocity
  0.10 m/s, isothermal (the reaction is only mildly exothermic), plug flow is fine.
- Catalyst: 6 mm spheres. Effective diffusivity inside the pellets 2e-7 m2/s
  (measured). Gas diffusivity 2e-5 m2/s.
- Kinetics: first order, rate constant 1.0 per second per unit pellet volume,
  measured on crushed catalyst (so intrinsic).
- We would like 95 % conversion.
EOF
