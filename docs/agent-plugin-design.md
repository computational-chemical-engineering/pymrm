# Design: a pymrm modelling plugin for coding agents

Status: APPROVED 2026-09-25 (two modes; verifier inherits the session model; persona files).
Implemented on branch `feature/claude-plugin`; section 10 lists where the build
departs from this draft.

## 1. Goal

Any pymrm user describes a process loosely. The agent talks with them about what
the model must answer and how detailed it must be, writes a specification the user
approves, builds a pymrm model to that specification, has it attacked by a separate
verifier, and reports what the model can and cannot say.

## 2. Decisions already taken

| Decision | Choice |
|---|---|
| Home | public `pymrm` repo, so API and guidance change in the same PR |
| Relation to `clearsheet` | neighbour: borrow its patterns, no dependency, nothing copied from the private repo |
| Deliverable | class-based `.py` module plus driver notebook (compact script for pointwise or ODE cases, style guide 2.1/2.2), plus a model card |
| Names | plugin `pymrm`; skills `conventions`, `build-model`, `verify-model`; Claude agent `model-verifier` |
| Single source | the style guide moves into the plugin; `docs/pymrm-model-style-guide.md` becomes a pointer stub |
| Portability | all substance in Agent Skills (open SKILL.md standard, read by Claude Code, Codex CLI, Gemini CLI); per-tool wrappers stay thin |
| Worked examples | public material only (gallery, `tutorials`, `examples`, `pymrm-book`); never `pymrm-book-teacher` |

Facts that fix the layout (confirmed from the Claude Code docs on 2026-09-25): an
installed plugin is copied to `~/.claude/plugins/cache/...` and may not read paths
outside its own directory; a marketplace can live in the same repo with
`"source": "./plugins/pymrm"`; components are namespaced `pymrm:<name>`;
`claude plugin eval` runs cases from an eval directory given by `--eval-dir`.

## 3. Layout

```
pymrm/
  AGENTS.md                         any tool, working ON pymrm: points to the conventions skill
  .claude-plugin/marketplace.json   marketplace "pymrm", one plugin, source ./plugins/pymrm
  plugins/pymrm/                    everything a user's agent can read; nothing else ships
    .claude-plugin/plugin.json
    README.md                       install for Claude Code, Codex CLI, Gemini CLI
    skills/
      conventions/
        SKILL.md                    short index: when to read which reference
        references/style-guide.md   git mv from docs/, scope widened, private-repo mention removed
        references/pitfalls.md      the measured traps, each with symptom, cause, fix, test name
        references/api-map.md       which pymrm function for which job; how to read the installed source
        exemplars/*.py              3 to 4 small models in house style, each with its own checks
      build-model/
        SKILL.md                    the workflow, its gates and its stopping rule
        references/elicit.md        the questions, and when to stop asking
        references/fidelity.md      phenomena, deciding groups, criteria, the ladder
        references/structures.md    structure codes S1 to S13 -> pymrm ingredients -> exemplar
        references/spec-template.md
        references/model-card-template.md
      verify-model/
        SKILL.md                    how to attack a model; verdict per spec assertion
        references/attack-list.md   defect classes measured in the gallery, hunt them by name
    agents/model-verifier.md        Claude wrapper: frontmatter + `skills: [verify-model]`
    evals/                          `claude plugin eval` cases (the harness hides them from the agent under test)
  test/test_pitfalls.py             one test per trap; fails if an API change makes a trap untrue
  docs/pymrm-model-style-guide.md   stub pointing to the new location
```

Why the exemplars are bundled rather than pointed to: `examples/` and `tutorials/`
sit outside the plugin directory and use legacy names (`Jac_const`, `g(...)`). A
few short exemplars written to the style guide, run in CI, are cheaper than
teaching the agent which legacy patterns to ignore. For API truth the skill tells
the agent to read the INSTALLED pymrm source (`python -c "import pymrm;
print(pymrm.__file__)"`), so the docstrings match the user's version. The gallery
is linked by URL for further examples.

## 4. The workflow (`build-model`)

Each phase ends with a written artefact. Two gates need the user.

0. **Environment.** pymrm installed and which version; if older than the plugin's
   minimum, say so.
1. **Elicit.** The question the model must answer and the decision that hangs on
   it; outputs and their required accuracy; operating range; data the user has;
   time budget. State the stopping rule now: what "done" means and when to park.
2. **Scope and fidelity.** List candidate phenomena. Estimate the groups that decide
   each (Pe, Da, Thiele, Biot, Mears, Weisz-Prater, ...) with labelled parameter
   values. Choose a rung on the ladder. For every excluded phenomenon, state what
   it would change and roughly by how much.
3. **Specification. GATE: user approves before any code.** Equations; each bc as
   the pymrm dict with the physical equation beside it; every parameter with a
   source label (user-given, literature with reference, correlation, assumed);
   assumptions; the validation plan with numbered assertions, chosen before the
   model and including at least one check that can fail and one headline computed
   by a second independent route.
4. **Implement.** Map to structure codes, copy the nearest exemplar, substitute
   the physics, follow `conventions`. Copying an exemplar does not copy its checks:
   rebuild them for the new physics.
5. **Self-check.** Refine every axis that carries error (grid and time step),
   report observed orders; root-find thresholds and extrema instead of sampling;
   reported numbers come from deterministic solves, not warm-start chains.
6. **Verify.** A separate context runs `verify-model` on the spec and the code,
   not on the builder's narrative. Verdict per assertion: met / not met /
   insufficient evidence / blocked. One fix round, then re-verify; if still not
   met, report it as such rather than iterate.
7. **Report. GATE: the user sees the model card.** Results; what was checked and
   how; which conclusions rest on assumed parameters; validity limits; and what
   the model does NOT establish.

Tool-neutral wording rule: skill text never names a tool-specific function. It
says "ask the user", "run the verifier in a separate context (a subagent if your
tool has one, otherwise a fresh session with this brief)".

## 5. The verifier

- Portable instructions in `verify-model`; the Claude wrapper
  `agents/model-verifier.md` adds only: read-only tools plus Bash to run code,
  `model: inherit`, and the skill preload.
- Inputs: the approved spec, the code, the self-check output. Not the builder's
  conclusions.
- Method: attack the baseline, not only the inputs. Rerun the numbers itself;
  check every boundary condition against its physical statement; refine grids;
  test a limit the builder did not test; hunt the defect classes by name.
- Output: verdict per spec assertion, with the reproducing command, and a list
  of what the evidence does not establish.

## 6. Pitfalls to move into `conventions/references/pitfalls.md`

Each gets symptom, cause, fix, and a test in `test/test_pitfalls.py`:

1. bc normal is OUTWARD, so the sign of `a` flips meaning between the two ends.
   The current style guide's own Danckwerts comment (`D * dc/dx + v * c = v *
   c_in`) is wrong for the left end; the dict is right. Fix the comment in the move.
2. Diffusivity jump: harmonic mean at the face (arithmetic converges at first
   order only).
3. `NumJac((n,))` on a single-field 1-D problem builds a dense Jacobian.
4. `axes_diagonals=[0]` on a 1-D shape drops the diagonal and converges to a
   different answer.
5. Changing boundary values: assemble once, use `shapes_d`.
6. Pure outflow boundary: pymrm has none; the Neumann outflow extrapolates to the
   face; the gallery workaround.
7. `newton` stops on an absolute step norm: scale mixed-unit state vectors.
8. Outlet values: read through `compute_boundary_values` or the face, never off
   the last cell centre (`v*C_N` is O(h): 8.9e-3 at n=8, 7.8e-5 at n=800, and a
   mass balance written with it fails to close).

Material that goes elsewhere: the gallery's structure codes S1 to S13 go into
`build-model/references/structures.md`; the D1.1 ladder goes into `fidelity.md`
(shifts against rung 1: +0.17 %, -4.74 %, +15.69 %, -8.94 %; resolving the
particle moves the answer +21.4 % from the film-only model, so stopping at the
film model leaves you further from the particle model than plug flow was) with
the warning that a passed rate criterion does not bound a sensitive output; the
"check that cannot fail" (B1.6: an identity residual of 1e-11 blind to a Newton
residual of 18) and "break something on purpose" go into the verifier's
attack list.

## 7. Evaluation

Cases live in `plugins/pymrm/evals/` (see section 10). Three kinds:

**A. Trap tasks**, one per pitfall, each with a measured wrong answer. Graded by
number (regex on printed output against a tolerance) plus a code check where the
trap is structural.

| Case | Task | Wrong answer it catches |
|---|---|---|
| T1 | slab, prescribed flux entering at the RIGHT end | sign of `a`, profile mirrored |
| T2 | composite slab, D jumps 10x | arithmetic face mean, first-order error |
| T3 | nonlinear reaction-diffusion, one field, n=400 | `NumJac((n,))` dense, or `axes_diagonals=[0]` |
| T4 | tanks in series, outlet concentration | Neumann outflow, 5.9% at N=2 |
| T5 | nonisothermal pellet, T in K and c in mol/m3 | unscaled newton stops early |
| T6 | open tube, outlet conversion and mass balance | outlet read off the last cell |

**B. Held-out gallery pages.** The model statement as a paragraph, no code, no
link; score against the page's `agreement.json` (headline metrics only, not all).

| Page | Structure | Metrics pinned |
|---|---|---|
| A2.1 Danckwerts boundary conditions | S3, S4 | 29 |
| B1.1 Thiele, Weisz-Hicks pellet (multiplicity) | S3 | 7 |
| A3.15 Graetz-Nusselt | S6 | 22 |
| B3.2 grain model | S8, S12 | 15 |
| D2.2 Van Welsenaere-Froment runaway | S2, two routes | 7 |
| J1.5 LDF breakthrough | S4, S5 | 5 |

**C. Vague-spec tasks**, graded by an LLM judge against a rubric: asked for the
purpose, decision and range; estimated the deciding groups with numbers; chose a
fidelity and justified it; labelled every parameter; wrote a spec before code;
ran a separate verifier; the model card says what is not established. Four
prompts, for example "Will my methanation bed run away?", "Is internal diffusion
limiting in my pellets?", a membrane sizing question, and one where the right
answer is that a criterion settles it and no PDE model is needed. The eval
harness has no live user, so each case carries a persona file with the user's
answers; the agent must write down the questions it would ask before it reads the
persona.

Runs: every case on Opus and Sonnet, 3 runs each. A no-plugin baseline arm only
for A, to show the traps are real for an unaided agent.

## 8. Order of work

1. Move the style guide, write `pitfalls.md` and `test/test_pitfalls.py`
   (tests first: each trap is demonstrated numerically before it is written down).
2. `conventions` skill, exemplars, CI job that runs the exemplars.
3. `build-model` and `verify-model` skills, Claude wrapper, marketplace files.
4. Eval cases; run on Opus and Sonnet; fix what fails.
5. `AGENTS.md`, README section; Codex and Gemini wrappers only after an eval case
   has run on each.

## 9. Open questions for the user

1. **Two modes?** Recommended: `estimate` (phases 1 to 2 and a short report: do
   you need a model at all, which groups decide it) and `model` (all phases). A
   verifier always runs in `model` mode. Without this, every question costs a
   full build.
2. **Verifier model.** Recommended: `model: inherit`, so it works for users
   without Opus; the Sonnet eval runs tell us whether that is good enough.
3. **Eval budget.** A 6 x 2 x 3 x 2 = 72 sessions, B 6 x 2 x 3 = 36, C 4 x 2 x 3 =
   24: about 130 sessions for a full pass. Recommended: full pass once before the
   PR, then only the cases a change touches.
4. **Persona files for the vague-spec cases** (section 7C): acceptable, or do you
   want those graded by hand in a live session?

## 10. As built (2026-09-25)

- **Eval location.** `claude plugin eval` only accepts an eval directory inside
  the plugin, and its docs state that a run cannot read the eval directory. The
  cases therefore live in `plugins/pymrm/evals/`; `scripts/run_plugin_evals.sh`
  runs them.
- **Eval sandbox.** Bash in an eval run is sandboxed and cannot read `$HOME`, so
  the runner installs pymrm non-editable into `/tmp/pymrm-eval-venv` and puts it
  first on `PATH`. Linux needs `bubblewrap` and `socat`.
- **Held-out pages: five, not six.** A3.15 was dropped: its only target that a
  paragraph can pose is the textbook Nu = 3.657, which an agent can recall.
- **Verifier model.** Plugin agents do not accept `model: inherit`; the agent file
  omits `model`, which inherits the session model.
- **Pitfall numbers were re-measured** on pymrm 2.4.6 and differ from the gallery
  write-ups where the test problem differs (for example P6: 3.85 % at N = 2 here,
  5.9 % on a different metric in the gallery). `pitfalls.md` quotes the tests.
- **Two extra rules in `build-model`:** a user who already gave a complete
  specification is not re-interviewed; a non-interactive run writes its
  questions to `questions.md` first and labels everything else assumed.
- **Found while building:** the style guide's minimal template used
  `NumJac((n_x,))` (dense) and a flux bc labelled like a Dirichlet one; its
  Danckwerts comment had the wrong sign at the left end. All fixed in the move.
  `NumJac(..., axes_diagonals=[0])` on a 1-D shape still returns a wrong
  Jacobian; worth fixing in pymrm itself (raise an error) rather than only
  documenting it.
- **Skill names stay short** (decided 2026-09-25). Claude Code always namespaces
  plugin skills (`/pymrm:build-model`), and Gemini CLI namespaces extension
  skills, so a `pymrm-` prefix would only stack. Name clashes arise only when the
  folders are copied into a flat skills directory. Step 2 of the order of work
  therefore ships the same folder as a Gemini extension (`gemini-extension.json`)
  and, if its skills are namespaced, a Codex plugin, both tested by an eval run;
  the flat copy stays a fallback with a clash warning in the README.
- **Codex and Gemini packaging** (tested 2026-09-27, Codex CLI 0.157.1, Gemini
  CLI 0.61.0). No extra manifests were needed. Codex reads
  `.claude-plugin/marketplace.json` and `plugin.json` as they are
  (`codex plugin marketplace add computational-chemical-engineering/pymrm`) and
  namespaces the skills (`$pymrm:build-model`). Gemini cannot install an
  extension from a subfolder, but `gemini skills install <repo> --path
  plugins/pymrm/skills` installs all four skills; they are not namespaced, and a
  reinstall silently overwrites a skill of the same name. Decided with the
  maintainer: keep the short names and document the clash check and
  `--scope workspace`. The verifier agent is Claude-only: a Codex plugin does not
  load `agents/` (a TOML agent works only from `~/.codex/agents/`), and Gemini's
  agent loader rejects the Claude file (`tools` must be a list, `skills` is an
  unknown key). Both tools use the fallback in `build-model` (a subagent or a
  fresh session pointed at `verify-model`). Not yet done: a model-building run
  in Gemini (no account logged in on the test machine).

## 11. Next phase (approved 2026-09-25)

Evidence behind it:
- In the gallery, about 39 of 47 NumJac pages already differentiate only local
  terms, with transport from operators. The 8 full-residual pages (A4.1, A4.2,
  A4.3, A4.4, A4.9, J3.1, J3.4, J3.5) all have fluxes nonlinear in the state.
  What goes unused is the coupling and boundary API: `shapes_d`,
  `update_csc_array_indices` and `construct_interface_matrices` appear in no
  page, `compute_boundary_values` in 12 of 82.
- A two-field (c, T) sphere pellet: analytic transport Jacobian plus NumJac on
  the source takes 1.05 ms per residual and Jacobian at n = 10 000, full-residual
  NumJac takes 1.86 ms; identical answers and iterations.

Decisions: the default assembly style is the maintainer's (operators for
transport, NumJac for local terms, Jacobian as a sum); full-residual NumJac is
recommended when fluxes are nonlinear in the state. Profiles: default,
efficiency, flexibility, readability, teaching. Pattern idioms may be distilled
from the maintainer's private repositories in new words and new public
exemplars, never copied.

Phases, each closed by evals:
1. Baseline: current eval suite on Opus and Sonnet.
2. `conventions/references/assembly-styles.md`, a CI-tested pair of the same
   model in both styles, profiles recorded in `brief.md` and readable from a
   project preferences section; evals for profile adherence.
3. `consult` mode (authoritative recommendation, real alternatives with
   trade-offs, no code unless asked) and `build-model/references/reactor-selection.md`.
4. `model-patterns` skill: monolithic multi-domain coupling (membrane reactor),
   pressure-velocity coupling, nested scales, segregated versus monolithic; one
   tested exemplar and one eval case per pattern.
5. Automated skill optimisation (CORAL-type or skill-refinement tools) against the
   suite, if phase 1 shows room.
6. Codex and Gemini packaging (section 10).

## 12. Progress and measurements (2026-09-26)

**Eval environment.** The first baseline (2026-09-25, $72) ran without pymrm:
the eval sandbox hides `$HOME`, `/tmp` and `/var/tmp`, and exposes only the PATH
directories of the home tree. `scripts/run_plugin_evals.sh` now builds a venv
whose packages live inside `bin/` with a wrapper `python`; the `z-env-probe` case
fails when pymrm is not importable. Transcripts are kept (`--keep-temp`).

**Measured with pymrm available** (one run per case):
- Opus passes all pitfall and held-out cases with AND without the plugin: on
  well-posed problems the plugin's value for Opus is process, not correctness.
- Sonnet cost 4 to 8 times Opus per task. Turn analysis: most of it was the
  full model-mode workflow (spec, verifier subagent, model card) on one-number
  calculations, verifier self-inflicted errors, fitting an asymptote instead of
  deriving it, and polling background runs. After the skill fixes: $22 to
  $9.97 for the same cases, Python runs per case 18 to 7.5, errors 5.8 to 2.1,
  pitfall tasks $0.26 to $0.42 (Opus $0.20 to $0.30).
- Opus's repeated errors: absolute residual thresholds, root finding without a
  verified bracket, scripts that cannot import the model, assumed packages,
  SciPy tolerance limits. Now `numerics.md` and `pymrm.checks`.
- Sonnet D2.2: two "independent" routes agreed on a model that ignored the
  stated constant density. An independent route checks numerics, not the reading
  of the problem (fix pending: map each stated assumption to its code line).

**Library changes (branch feature/api-guards, merged into this branch):**
P4 stencil fixed, P3 warning, newton `rtol` and `step_norm`, `describe_bc`,
`pymrm.checks`, `{"outflow": True}` marker; reviewed twice by `think`, 390
tests, every regression test fails on the previous code. Found on the way:
with `shapes_d` the bc dictionary's `d` is a coefficient on the external vector.

**Plugin additions:** `model-patterns` skill with three tested exemplars
(membrane reactor, Darcy pressure-velocity, nested particles with Schur
complement); format rule (flat executed notebook for simple or didactic models,
modules plus driver notebook for full models) with `pellet_teaching.ipynb`;
assembly styles and profiles; consult mode; reactor selection.

**Proposed for the gallery (not done; the gallery is paused):** page = folder
with `model.py` (importable, tested) + `index.ipynb` + `data/`; the bootstrap
fetches `model.py` like `gallery_utils.py`; pin pymrm and fetch from a tag, not
`main`; migrate pages when they are next touched.

**Like-for-like comparison with outcome graders** (2026-09-26, one run per
case; file-protocol graders scored only in the plugin arm):
- Opus: equal outcomes with and without the plugin on 9 of 12 cases (vague
  c2-c4, consult k1-k3, patterns m1-m2, teaching p1). The plugin helps on m3
  (particle-bed coupling actually solved) and p2 (efficient code); c1 exceeds
  the 60-minute limit in both arms. The plugin roughly doubles Opus's cost and
  turns per task (spec, checks, verification).
- Sonnet on the coupled-structure cases: the plugin helps on m2 (conversion
  right, wrong without) and m3 (coupling solved); m1 with the plugin missed one
  value that the no-plugin run got (single runs; possibly noise).
- Reading: for Opus the plugin's value on these tasks is the audit trail and
  robustness on coupled structures, not correctness on well-posed problems; for
  Sonnet it is also correctness. A multi-run baseline is needed before quoting
  rates.

**Notebook maths.** The portable subset is in style guide 3.1 and checked by
`scripts/check_notebook_math.py` (KaTeX parse). Across the 82 gallery pages it
flags 46 pages, including 21 expressions KaTeX cannot parse (VS Code), mostly
two `\tag` in one block. The manual check of `docs/math-render-test.ipynb`
(2026-09-27) found one failure per renderer: `\eqref` on GitHub and `\label` in
VS Code; blank lines around `$$`, a bare `align`, `\\` and `*` in inline maths
and `|` in tables all rendered. The checker now reports only confirmed failures
(and KaTeX parse errors) as errors, the rest as warnings. Colab rendered
everything except `\eqref`, which gives no equation number there either.

**Three-run targeted comparison and hard cases** (2026-09-26):
- After replacing LLM numeric judges with deterministic range regexes (the
  judge had failed correct numbers in 3 of 3 runs), no measured case shows a
  correctness difference between with and without the plugin: m1-m3, a3, and
  the three hard cases (adsorption front width, ternary Maxwell-Stefan signs,
  hidden internal diffusion) are solved correctly by Opus and Sonnet in both
  arms. The one repeatable difference is p2 (efficiency sweep: refinement study
  and cold-start checks present with the plugin, 2/3 against 0/3); m3 differs
  only in style (monolithic or Schur with the plugin, particle nested in the
  bulk ODE without, both exact). Sonnet D2.2 misreads "constant density" in 1
  to 2 of 3 runs in either arm.
- Cost per task with the plugin is 1.2 to 2 times higher for Opus and up to 17
  times for Sonnet on a vague case (h3: $4.20 against $0.24, same answer).
- Reading: on problems of this size, current models are capable pymrm modellers
  unaided. The plugin's measurable value is process (spec, checks, verification,
  audit trail), the maintainer's conventions, and teaching output; its cost is
  real. Candidates: make the full model-mode workflow opt-in, keep
  `conventions` and `model-patterns` light, and look for value in longer,
  interactive work that single-shot evals do not capture.

**Slim-down (2026-09-27).** `build-model` defaults to direct mode; the full
workflow runs on request. First test: all cases correct, but the agent started
the full workflow unasked in 4 of 14 runs, triggered by eval prompts that said
"treat your specification as approved". After making direct mode explicit for
unattended runs: Opus c3 $2.56 to $0.53, c4 $2.76 to $0.70, Sonnet h3 $2.90 to
$0.66, all correct and in direct mode; one Sonnet run still read the old prompt
wording as a request for documentation (prompts now reworded). In that full-mode
run the verifier caught a real unit-conversion bug. Open: Sonnet h2 misread the
ternary problem in 1 of 3 runs (two routes agreed on the misreading), like D2.2.
