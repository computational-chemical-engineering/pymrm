"""Record every ``pymrm.newton`` call made by a script or notebook.

``install(logdir, tag)`` replaces ``pymrm.newton`` (and ``pymrm.solve.newton``)
by a wrapper that calls the original and appends one JSON line per call to
``<logdir>/<tag>.jsonl``. The solution ``x`` is appended as raw float64 to
``<logdir>/<tag>.x``; the record holds its offset. Each record also carries
the backward error of ``x`` from :func:`pymrm.checks.residual_check`, which
costs one extra evaluation of the residual function per call.

Install before the code under test imports ``newton``.
"""

import inspect
import json
import os
import time

import numpy as np


def _describe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if callable(value):
        return getattr(value, "__name__", type(value).__name__)
    if isinstance(value, np.ndarray):
        return f"array{value.shape}"
    return type(value).__name__


def _caller():
    import pymrm
    here = os.path.abspath(__file__)
    package = os.path.dirname(os.path.abspath(pymrm.__file__)) + os.sep
    for frame in inspect.stack()[2:]:
        name = os.path.abspath(frame.filename)
        if name != here and not name.startswith(package):
            return f"{os.path.basename(frame.filename)}:{frame.lineno}:{frame.function}"
    return "?"


def install(logdir, tag):
    import pymrm
    import pymrm.solve
    from pymrm.checks import residual_check

    original = pymrm.solve.newton
    if getattr(original, "_logged", False):
        return
    signature = inspect.signature(original)
    os.makedirs(logdir, exist_ok=True)
    rec_path = os.path.join(logdir, f"{tag}.jsonl")
    x_path = os.path.join(logdir, f"{tag}.x")
    open(rec_path, "w").close()
    open(x_path, "wb").close()
    state = {"idx": 0, "offset": 0}

    def newton(function, initial_guess, *args, **kwargs):
        idx = state["idx"]
        state["idx"] += 1
        bound = signature.bind(function, initial_guess, *args, **kwargs)
        options = {k: _describe(v) for k, v in bound.arguments.items()
                   if k not in ("function", "initial_guess")}
        record = {"idx": idx, "caller": _caller(), "n": int(np.size(initial_guess)),
                  "shape": list(np.shape(initial_guess)), "options": options}
        t0 = time.perf_counter()
        try:
            result = original(function, initial_guess, *args, **kwargs)
        except Exception as exc:  # recorded, then re-raised unchanged
            record.update(time=time.perf_counter() - t0, raised=f"{type(exc).__name__}: {exc}"[:300])
            with open(rec_path, "a") as fh:
                fh.write(json.dumps(record) + "\n")
            raise
        record["time"] = time.perf_counter() - t0
        for key in ("success", "nit", "nfev", "nfact", "status", "message", "step_norm"):
            if key in result:
                value = result[key]
                record[key] = value.item() if isinstance(value, np.generic) else value
        x = np.asarray(result.x, dtype=np.float64)
        record["x_offset"] = state["offset"]
        record["x_finite"] = bool(np.all(np.isfinite(x)))
        with open(x_path, "ab") as fh:
            fh.write(x.tobytes())
        state["offset"] += x.size
        fargs = bound.arguments.get("args", ())
        solver = bound.arguments.get("solver")
        if solver == "splu" or not record["x_finite"]:
            record["backward_error"] = None
        else:
            try:
                with np.errstate(all="ignore"):
                    check = residual_check(lambda u: function(u, *fargs), x.reshape(np.shape(initial_guess)))
                record["backward_error"] = check["backward_error"]
                record["max_abs_residual"] = check["max_abs_residual"]
            except Exception as exc:
                record["backward_error"] = None
                record["check_error"] = f"{type(exc).__name__}: {exc}"[:200]
        with open(rec_path, "a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        return result

    newton._logged = True
    newton.__doc__ = original.__doc__
    pymrm.newton = newton
    pymrm.solve.newton = newton
