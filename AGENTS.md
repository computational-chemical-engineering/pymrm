# Instructions for coding agents working on pymrm

This file is for agents (Claude Code, Codex, Gemini CLI and others) that edit the
pymrm library itself. Agents that BUILD MODELS with pymrm should use the skills
in `plugins/pymrm/` instead; see `plugins/pymrm/README.md`.

## Conventions

The modelling conventions and the known API pitfalls live in
`plugins/pymrm/skills/conventions/`:

- `references/pitfalls.md`: API behaviour that has produced wrong answers, ids
  P1 to P8.
- `references/style-guide.md`: the house style for models, tutorials and examples.
- `references/api-map.md`: which function does which job.

## Keeping guidance and API together

Each pitfall is pinned by a test in `test/test_pitfalls.py`, and the exemplar
models the skills tell agents to copy are run by `test/test_plugin_exemplars.py`.

- If you change behaviour that a pitfall describes, a pitfall test will fail.
  Update `pitfalls.md` (and the style guide, if it quotes the behaviour) in the
  same change, then the test.
- If you change a public signature, check `api-map.md` and the exemplars.
- If you fix a pitfall (for example make `NumJac` reject `axes_diagonals` on a 1-D
  shape), delete the entry and its test rather than leaving stale advice.

## Development

```bash
pip install -e . pytest flake8
pytest test/                   # unit tests, pitfall tests, exemplar checks
flake8 src/                    # CI lints src/ only; keep new test files clean too
```

Public docstrings are NumPy style; keep them accurate, agents read them as the
API reference.
