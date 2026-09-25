# Profiles: what the model is optimised for

A profile steers choices the physics leaves open. Record it in `brief.md`. If the
user does not name one, use `default`, unless project instructions (for example
a "pymrm modelling preferences" section in `AGENTS.md`, `CLAUDE.md` or the
repository README) name a different default. A user's explicit request always
wins over the project default.

The profile never relaxes correctness: the pitfalls, the validation plan and the
verifier apply to every profile.

| | default | efficiency | flexibility | readability | teaching |
|---|---|---|---|---|---|
| Purpose | a model the user will run and extend | many solves: sweeps, optimisation, fitting, large grids | a model that will grow: more species, phases, closures | a model others must read and trust | a model that explains the method |
| Assembly (see `assembly-styles.md`) | A; B for multi-domain | A; constant parts factorised once (`splu`, `solver="splu"`); B only with constant blocks placed once | A or hybrid; closures (kinetics, properties, correlations) passed in as callables | A | A, every Jacobian term a named matrix; B shown explicitly when the model has domains |
| Code form | class-based module + driver notebook (style guide 2.2) | class-based; vectorised; no Python loops over cells; timing reported | class with injected closures and a parameter object; unit tests per closure | class-based, short methods named after physics, few abstractions | notebook-first; small steps with markdown between them; derivation of each discrete term |
| Grid and solver | uniform, refined until converged | non-uniform grids where profiles are steep; minimal n meeting the accuracy target; warm starts only for speed, never for reported numbers | as default | as default | uniform; the grid-refinement study is part of the lesson |
| Comments | physical equation beside each bc and balance | same, plus cost notes where a choice is made for speed | interfaces documented: what a closure receives and returns | physical meaning over mechanics | why each step, plus a sparsity plot or a printed small Jacobian |
| Extra output | model card | timing and scaling table (evaluations, solves, wall time vs n) | a short "how to extend" section | none | exercises or questions for the reader, if asked |

Combining: a user may give two priorities ("teaching, but it must run a
parameter sweep"). Take the first as the profile and apply the second only
where it does not conflict, and say which choices you made for which reason.
