#!/bin/bash
set -e
cat > persona.md <<'EOF'
Answers from the researcher:
- Decision: whether to order a smaller pellet size (1.5 mm) for our pilot unit.
- Pellets: 3.0 mm diameter spheres, pellet density 1500 kg m-3, porosity 0.5,
  mean pore diameter 10 nm (from N2 physisorption). Tortuosity unknown.
- Measured rate on these pellets: 2.0e-3 mol per kg catalyst per s, at 600 K,
  5 bar, 10 mol % reactant (molar mass 28 g mol-1) in an inert gas.
- Kinetics: probably first order, not sure.
- A factor of two in accuracy is fine; we just need to know if it matters.
EOF
