#!/bin/bash
# Run the coding-agent plugin's eval suite (plugins/pymrm/evals) with `claude plugin eval`.
# Execute from the repository root. Extra arguments go to `claude plugin eval`, e.g.
#   scripts/run_plugin_evals.sh --case 'a*' --runs 1 --model claude-sonnet-5
#
# Eval runs execute Bash inside Claude Code's OS sandbox. It hides $HOME, /tmp and /var/tmp,
# and a venv inside the plugin folder makes it too large to scan, so pymrm is installed
# (non-editable) into a venv under /opt and put first on PATH. Create it once with:
#   sudo mkdir -p /opt/pymrm-eval-venv && sudo chown "$USER" /opt/pymrm-eval-venv
# Override the location with PYMRM_EVAL_VENV. Linux needs bubblewrap and socat.
# The z-env-probe case checks that pymrm is importable inside the sandbox; run it first:
#   scripts/run_plugin_evals.sh --tag env --runs 1 --ablation none
set -euo pipefail

EVAL_VENV="${PYMRM_EVAL_VENV:-/opt/pymrm-eval-venv}"
PYTHON="${PYTHON:-python3}"

for tool in bwrap socat; do
    command -v "$tool" >/dev/null || { echo "missing '$tool' (needed by the eval sandbox)"; exit 1; }
done

if [ ! -x "$EVAL_VENV/bin/python" ] || [ "${PYMRM_EVAL_REINSTALL:-0}" = "1" ]; then
    "$PYTHON" -m venv "$EVAL_VENV"
    "$EVAL_VENV/bin/pip" install -q --upgrade pip
    "$EVAL_VENV/bin/pip" install -q . matplotlib
fi
"$EVAL_VENV/bin/python" -c "import pymrm; print('pymrm for evals:', pymrm.__file__)"

PATH="$EVAL_VENV/bin:$PATH" claude plugin eval plugins/pymrm \
    --allow-tools Bash Write Edit \
    --scaffold --trust-plugin --no-publish --keep-temp "$@"
# --keep-temp keeps each run's transcript at /tmp/claude-eval-*/out/trace.jsonl
# (tracePath in the --json output); delete those directories when done.
