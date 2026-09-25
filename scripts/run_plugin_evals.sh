#!/bin/bash
# Run the coding-agent plugin's eval suite (plugins/pymrm/evals) with `claude plugin eval`.
# Execute from the repository root. Extra arguments go to `claude plugin eval`, e.g.
#   scripts/run_plugin_evals.sh --case 'a*' --runs 1 --model claude-sonnet-5
#
# Eval runs execute Bash inside Claude Code's OS sandbox, which cannot read your home
# directory. pymrm is therefore installed (non-editable) into a venv outside $HOME
# and put first on PATH. Linux needs bubblewrap and socat for the sandbox.
set -euo pipefail

EVAL_VENV="${PYMRM_EVAL_VENV:-/tmp/pymrm-eval-venv}"
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
    --scaffold --trust-plugin --no-publish "$@"
