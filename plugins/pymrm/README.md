# pymrm skills for coding agents

Skills that let a coding agent turn a loose description of a process into a
scoped, verified pymrm model, and tell you honestly what that model can and
cannot say.

| Skill | What it does |
|---|---|
| `build-model` | Asks what the model must decide, estimates which phenomena matter (Thiele, Weisz-Prater, Mears, Peclet, Hatta, ...), writes a specification for your approval, builds the model, has it verified, and writes a model card. Has a quick `estimate` mode for "does this matter at all?" questions. |
| `verify-model` | Attacks a finished model against its specification in a fresh context and returns a verdict per check. |
| `conventions` | pymrm house style, known API pitfalls (each pinned by a test), and tested exemplar models. Useful on its own whenever an agent writes pymrm code. |

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
