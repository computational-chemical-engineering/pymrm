#!/bin/bash
set -e
cat > persona.md <<'EOF'
Answers from the plant engineer:
- Decision: the highest coolant (oil) temperature we can use without thermal
  runaway, with a safety margin. Currently 523 K.
- Tubes: inner diameter 25 mm, length 3 m. Catalyst: 3 mm spheres, bed voidage
  0.40, bulk density 1100 kg m-3.
- Feed: H2:CO2 = 4:1, 5 bar, 523 K, superficial velocity 1.0 m s-1 at inlet
  conditions. Gas heat capacity about 30 J mol-1 K-1.
- Reaction CO2 + 4 H2 -> CH4 + 2 H2O, heat of reaction -165 kJ per mol CO2.
- Kinetics (from our supplier, fitted 500-600 K): rate per kg catalyst
  r = k p_CO2 with k = 2.0e-3 mol kg-1 s-1 bar-1 at 523 K and activation
  energy 80 kJ mol-1. Nothing better available.
- Overall wall heat transfer coefficient: 150 W m-2 K-1 (our estimate).
- Effective radial conductivity of the bed: we do not know.
- Steady state is enough. We need a first answer today: the safe coolant
  temperature to within about 10 K. If a more detailed model (for example
  radial gradients) could change it by more than that, tell us and by roughly
  how much, rather than building it now.
EOF
