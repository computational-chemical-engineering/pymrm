# pymrm skills for coding agents

Skills that help a coding agent model chemical-engineering processes with pymrm
in the house style: the pymrm conventions and known pitfalls, tested exemplar
models, checkers for models and notebooks, coupling patterns, and an optional
independent reviewer for decision-grade work.

What to expect: current models (Claude Opus and Sonnet) already build correct
pymrm models for well-posed problems without these skills, because they read the
installed library's source and docstrings. The skills add consistency with the
pymrm house conventions (operator-sum Jacobians, monolithic coupling, the model
and notebook formats), teaching-format notebooks, proportionate checks, and a
documented, independently verified workflow when you ask for one. See
`docs/agent-plugin-design.md` in the pymrm repository for the measurements.

| Skill | What it does |
|---|---|
| `build-model` | Builds and checks a model from a process description (default), gives quick estimates ("does this matter at all?"), discusses modelling choices, and on request runs a full workflow: specification for your approval, independent verification, model card. |
| `verify-model` | Attacks a finished model against its specification in a fresh context and returns a verdict per check. |
| `conventions` | pymrm house style, known API pitfalls (each pinned by a test), tested exemplar models, a run-time pitfall checker and a notebook maths checker. Useful on its own whenever an agent writes pymrm code. |
| `model-patterns` | Monolithic coupling of domains and phases, pressure-velocity coupling, nested scales, with tested exemplars. |

The skills follow the open Agent Skills format (`SKILL.md`), so the same files
work in several agent tools. pymrm itself must be installed in the Python
environment the agent uses: `pip install pymrm`.

## Claude Code

```text
/plugin marketplace add computational-chemical-engineering/pymrm
/plugin install pymrm@pymrm
```

Then describe your process, for example "I want to know whether internal
diffusion limits my 3 mm pellets", or invoke `/pymrm:build-model` directly. The
plugin also provides the `pymrm:model-verifier` subagent, which runs the
`verify-model` skill in its own context.

To try a local checkout: `claude --plugin-dir plugins/pymrm`.

## Other agent tools

Copy (or symlink) the three skill folders into the tool's skills directory:

```bash
# Codex CLI and Gemini CLI both read ~/.agents/skills
mkdir -p ~/.agents/skills
cp -r plugins/pymrm/skills/* ~/.agents/skills/
```

`build-model` asks for the verifier to run in a separate context. In tools with
subagents, point a subagent at the `verify-model` skill; otherwise start a fresh
session and ask it to verify with that skill. Dedicated Codex and Gemini agent
definitions are not included yet because they have not been tested.

## Maintaining

The pitfalls are tested in `test/test_pitfalls.py` and the exemplars in
`test/test_plugin_exemplars.py` of the pymrm repository, so an API change that
makes the guidance wrong fails CI. See `AGENTS.md` at the repository root.
