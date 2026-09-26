#!/bin/bash
# Run the coding-agent plugin's eval suite (plugins/pymrm/evals) with `claude plugin eval`.
# Execute from the repository root. Extra arguments go to `claude plugin eval`, e.g.
#   scripts/run_plugin_evals.sh --case 'a*' --runs 1 --model claude-sonnet-5
#
# Eval runs execute Bash inside Claude Code's OS sandbox. It hides $HOME, /tmp and /var/tmp,
# but exposes each directory on PATH (and the plugin folder). A normal venv therefore does
# not work: its bin/ is visible but its lib/ is not. This script builds a venv whose
# packages live INSIDE bin/, with bin/python a wrapper that sets PYTHONPATH, and puts that
# bin/ first on PATH. pymrm is installed non-editable (an editable install points at the
# hidden source tree). Rebuild after changing pymrm with PYMRM_EVAL_REINSTALL=1.
# Linux needs bubblewrap and socat. The z-env-probe case checks the setup; run it first:
#   scripts/run_plugin_evals.sh --tag env --runs 1 --ablation none
set -euo pipefail

EVAL_VENV="${PYMRM_EVAL_VENV:-$HOME/.cache/pymrm-eval-venv}"
PYTHON="${PYTHON:-/usr/bin/python3}"

for tool in bwrap socat; do
    command -v "$tool" >/dev/null || { echo "missing '$tool' (needed by the eval sandbox)"; exit 1; }
done

if [ ! -x "$EVAL_VENV/bin/python" ] || [ "${PYMRM_EVAL_REINSTALL:-0}" = "1" ]; then
    case "$EVAL_VENV" in */pymrm-eval-venv|*/.venv-eval) rm -rf "$EVAL_VENV" ;; esac
    "$PYTHON" -m venv "$EVAL_VENV"
    "$EVAL_VENV/bin/pip" install -q --upgrade pip
    "$EVAL_VENV/bin/pip" install -q . matplotlib
    pyver="$("$EVAL_VENV/bin/python" -c 'import sys; print(f"python{sys.version_info[0]}.{sys.version_info[1]}")')"
    real_python="$(readlink -f "$EVAL_VENV/bin/python")"
    mv "$EVAL_VENV/lib" "$EVAL_VENV/bin/lib"
    rm -f "$EVAL_VENV/lib64" "$EVAL_VENV/bin/python" "$EVAL_VENV/bin/python3" "$EVAL_VENV/bin/$pyver"
    for name in python python3; do
        printf '#!/bin/sh\n# eval-sandbox python: packages live inside this bin folder\nPYTHONPATH="%s${PYTHONPATH:+:$PYTHONPATH}" exec %s "$@"\n' \
            "$EVAL_VENV/bin/lib/$pyver/site-packages" "$real_python" > "$EVAL_VENV/bin/$name"
        chmod +x "$EVAL_VENV/bin/$name"
    done
fi
"$EVAL_VENV/bin/python" -c "import pymrm; print('pymrm for evals:', pymrm.__file__)"

PATH="$EVAL_VENV/bin:$PATH" claude plugin eval plugins/pymrm \
    --allow-tools Bash Write Edit \
    --scaffold --trust-plugin --no-publish --keep-temp "$@"
# --keep-temp keeps each run's transcript at /tmp/claude-eval-*/out/trace.jsonl
# (tracePath in the --json output); delete those directories when done.
