#!/usr/bin/env python3
"""Run the notebooks and scripts that call ``pymrm.newton`` and log every call.

    python run_corpus.py --out LOGDIR [--lib SRC] [--jobs N] [--only SUBSTR] ROOT...

Each ROOT is searched for ``*.ipynb`` and ``*.py`` files that mention
``newton``; files under ``_build``, ``_site``, ``queue_cases``, ``.ipynb_checkpoints``
and ``public-book`` copies are skipped. Notebooks run with
nbclient in their own directory with a first cell that installs the logger
(``newton_log.install``); nothing is written back. Scripts run with ``runpy``.
``--lib`` puts a pymrm source tree first on ``PYTHONPATH``, so a frozen
baseline can be compared with the working tree.

Run the roots from clean ``git worktree`` checkouts: notebooks may write
figures or caches next to themselves.

Writes ``LOGDIR/<tag>.jsonl`` and ``<tag>.x`` per file (see newton_log.py) and
``LOGDIR/summary.json`` with the run status of every file.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKIP = ("_build", "_site", "queue_cases", ".ipynb_checkpoints", "public-book",
        "__pycache__", "/build/", "check_model.py")


def discover(root):
    root = Path(root).resolve()
    hits = []
    for path in sorted(list(root.rglob("*.ipynb")) + list(root.rglob("*.py"))):
        text_path = str(path)
        if any(s in text_path[len(str(root)):] + "/" for s in SKIP):
            continue
        if path.suffix == ".py" and path.name in ("build_page.py", "newton_log.py", "run_corpus.py"):
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if re.search(r"\bnewton\b", text) and (path.suffix == ".ipynb" or "__main__" in text
                                                or "pymrm" in text):
            hits.append((root, path))
    return hits


def tag_of(root, path):
    rel = path.relative_to(root)
    return f"{root.parent.name}-{root.name}__" + "__".join(rel.with_suffix("").parts)


def run_one(root, path, out, env, timeout):
    tag = tag_of(root, path)
    pre = (f"import sys; sys.path.insert(0, {str(HERE)!r}); import newton_log; "
           f"newton_log.install({str(out)!r}, {tag!r})")
    t0 = time.perf_counter()
    if path.suffix == ".ipynb":
        code = (
            "import sys, nbformat\n"
            "from nbclient import NotebookClient\n"
            f"nb = nbformat.read({str(path)!r}, as_version=4)\n"
            f"nb.cells.insert(0, nbformat.v4.new_code_cell({pre!r}))\n"
            f"NotebookClient(nb, timeout={timeout}, kernel_name='python3', allow_errors=False,\n"
            f"    resources={{'metadata': {{'path': {str(path.parent)!r}}}}}).execute()\n"
        )
    else:
        code = f"{pre}\nimport runpy\nrunpy.run_path({str(path)!r}, run_name='__main__')\n"
    try:
        proc = subprocess.run([sys.executable, "-c", code], cwd=path.parent, env=env,
                              capture_output=True, text=True, timeout=timeout + 60)
        ok = proc.returncode == 0
        err = "" if ok else (proc.stderr.strip().splitlines() or ["?"])[-1][:300]
    except subprocess.TimeoutExpired:
        ok, err = False, "timeout"
    return {"tag": tag, "file": str(path), "ok": ok, "time": time.perf_counter() - t0, "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lib")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--only", default="", help="comma-separated substrings of file paths")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    only = [o for o in a.only.split(",") if o] or [""]
    files = [h for r in a.roots for h in discover(r) if any(o in str(h[1]) for o in only)]
    if a.list:
        for root, path in files:
            print(tag_of(root, path))
        return 0
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYDEVD_DISABLE_FILE_VALIDATION="1")
    if a.lib:
        env["PYTHONPATH"] = str(Path(a.lib).resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    results = []
    with ThreadPoolExecutor(a.jobs) as pool:
        futures = [pool.submit(run_one, r, p, out, env, a.timeout) for r, p in files]
        for fut in futures:
            res = fut.result()
            results.append(res)
            print(f"{'OK  ' if res['ok'] else 'FAIL'} {res['time']:7.1f} s  {res['tag']}"
                  + (f"  {res['error']}" if res["error"] else ""), flush=True)
    summary_path = out / "summary.json"
    old = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    old.update({r["tag"]: r for r in results})
    summary_path.write_text(json.dumps(old, indent=1))
    print(f"{sum(r['ok'] for r in results)}/{len(results)} files ran")
    return 0


if __name__ == "__main__":
    sys.exit(main())
