#!/usr/bin/env python3
"""Compare two corpus runs of ``run_corpus.py`` call by call.

    python compare.py BASE NEW [--xtol 1e-8] [--json OUT]

Calls are matched by file and call index. Reported: files whose run status
changed, calls whose ``success`` changed, calls whose ``x`` differs by more
than ``xtol`` relative to ``max |x|`` of the baseline, the change in Newton
iterations and function evaluations, and run time per file. With one
argument it summarises a single run, including the calls that report
``success=True`` with a backward error above 1e-8.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load(run, tag):
    recs = [json.loads(line) for line in (run / f"{tag}.jsonl").read_text().splitlines() if line]
    xs = np.fromfile(run / f"{tag}.x", dtype=np.float64) if (run / f"{tag}.x").exists() else None
    return recs, xs


def x_of(rec, xs):
    if xs is None or "x_offset" not in rec:
        return None
    return xs[rec["x_offset"]:rec["x_offset"] + rec["n"]]


def summarise(run):
    summary = json.loads((run / "summary.json").read_text())
    n_calls = n_success = 0
    false_success = {}
    for tag in sorted(summary):
        if not (run / f"{tag}.jsonl").exists():
            continue
        recs, _ = load(run, tag)
        n_calls += len(recs)
        for r in recs:
            n_success += bool(r.get("success"))
            be = r.get("backward_error")
            if r.get("success") and be is not None and be > 1e-8:
                false_success.setdefault(tag, []).append(be)
    ok = sum(v["ok"] for v in summary.values())
    print(f"{run}: {ok}/{len(summary)} files ran, {n_calls} newton calls, {n_success} successful")
    failed = [t for t, v in summary.items() if not v["ok"]]
    for t in failed:
        print(f"  FAILED FILE {t}: {summary[t]['error']}")
    print(f"success=True with backward error > 1e-8: "
          f"{sum(len(v) for v in false_success.values())} calls in {len(false_success)} files")
    for tag, bes in sorted(false_success.items()):
        print(f"  {tag}: {len(bes)} calls, backward error {min(bes):.1e} .. {max(bes):.1e}")


def compare(base, new, xtol):
    sb = json.loads((base / "summary.json").read_text())
    sn = json.loads((new / "summary.json").read_text())
    report = {"files": [], "lost": [], "gained": [], "x_changed": [], "ncalls": []}
    tot = {"nit_b": 0, "nit_n": 0, "nfev_n": 0, "calls": 0, "t_b": 0.0, "t_n": 0.0}
    for tag in sorted(sb):
        if tag not in sn:
            report["files"].append((tag, "missing in new run"))
            continue
        if sb[tag]["ok"] != sn[tag]["ok"]:
            report["files"].append((tag, f"ok {sb[tag]['ok']} -> {sn[tag]['ok']}: {sn[tag]['error']}"))
        tot["t_b"] += sb[tag]["time"]
        tot["t_n"] += sn[tag]["time"]
        if not (base / f"{tag}.jsonl").exists() or not (new / f"{tag}.jsonl").exists():
            continue
        rb, xb = load(base, tag)
        rn, xn = load(new, tag)
        if len(rb) != len(rn):
            report["ncalls"].append((tag, len(rb), len(rn)))
        for a, b in zip(rb, rn):
            tot["calls"] += 1
            tot["nit_b"] += a.get("nit", 0) or 0
            tot["nit_n"] += b.get("nit", 0) or 0
            tot["nfev_n"] += b.get("nfev", b.get("nit", 0)) or 0
            if bool(a.get("success")) and not bool(b.get("success")):
                report["lost"].append((tag, a["idx"], a.get("caller"), b.get("message")))
            if not bool(a.get("success")) and bool(b.get("success")):
                report["gained"].append((tag, a["idx"], a.get("caller"), a.get("message")))
            x1, x2 = x_of(a, xb), x_of(b, xn)
            if x1 is not None and x2 is not None and x1.size == x2.size:
                with np.errstate(all="ignore"):
                    ref = np.nanmax(np.abs(x1)) if x1.size else 0.0
                    diff = np.nanmax(np.abs(x1 - x2)) / max(ref, 1e-300) if x1.size else 0.0
                if not np.isfinite(diff) or diff > xtol:
                    report["x_changed"].append((tag, a["idx"], a.get("caller"), float(diff),
                                                a.get("success"), b.get("success")))
    print(f"calls compared: {tot['calls']}; Newton iterations {tot['nit_b']} -> {tot['nit_n']}; "
          f"function evaluations in new run {tot['nfev_n']}")
    print(f"total file time {tot['t_b']:.0f} s -> {tot['t_n']:.0f} s")
    for key in ("files", "ncalls", "lost", "gained"):
        print(f"\n{key}: {len(report[key])}")
        for item in report[key][:40]:
            print("  ", item)
    by_file = {}
    for tag, idx, caller, diff, s1, s2 in report["x_changed"]:
        by_file.setdefault(tag, []).append(diff)
    print(f"\nx changed by more than {xtol:g}: {len(report['x_changed'])} calls in {len(by_file)} files")
    for tag, diffs in sorted(by_file.items()):
        print(f"   {tag}: {len(diffs)} calls, max rel diff {max(diffs):.2e}")
    slow = []
    for tag in sorted(sb):
        if tag in sn and sb[tag]["time"] > 5:
            ratio = sn[tag]["time"] / sb[tag]["time"]
            if ratio > 1.2:
                slow.append((tag, round(sb[tag]["time"], 1), round(sn[tag]["time"], 1)))
    print(f"\nfiles more than 20% slower (base > 5 s): {len(slow)}")
    for item in slow:
        print("  ", item)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--xtol", type=float, default=1e-8)
    ap.add_argument("--json")
    a = ap.parse_args()
    runs = [Path(r) for r in a.runs]
    if len(runs) == 1:
        summarise(runs[0])
        return 0
    report = compare(runs[0], runs[1], a.xtol)
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
