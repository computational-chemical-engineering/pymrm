#!/usr/bin/env python3
"""Time and memory of newton's direct LU against ILU-BiCGSTAB at large size.

    python size_check.py [n]        # n x n cells, default 450 (about 2e5 unknowns)

A 2-D square with second-order kinetics, Dirichlet on all sides. Each solver
runs in a fresh process so the peak resident memory is its own.
"""

import subprocess
import sys

CODE = r"""
import json, resource, sys, time
import numpy as np
from pymrm import NumJac, construct_div, construct_grad, newton
n = int(sys.argv[1]); solver = sys.argv[2]
shape = (n, n, 1)
x_f = np.linspace(0.0, 1.0, n + 1)
bc = ({"a": 0.0, "b": 1.0, "d": 1.0}, {"a": 0.0, "b": 1.0, "d": 1.0})
jac = None; g0 = 0.0
for axis in (0, 1):
    grad, grad_bc = construct_grad(shape, x_f, bc=bc, axis=axis)
    div = construct_div(shape, x_f, nu=0, axis=axis)
    jac = (div @ -grad) if jac is None else jac + div @ -grad
    g0 = g0 + (div @ -grad_bc).toarray().reshape(-1)
jac = jac.tocsc()
numjac = NumJac(shape)
def fun(c):
    g_r, j_r = numjac(lambda c: 50.0 * c**2, c.reshape(shape))
    return g0 + jac @ c.reshape(-1) + g_r.reshape(-1), (jac + j_r).tocsc()
t0 = time.perf_counter()
res = newton(fun, np.ones(shape), solver=None if solver == "default" else solver,
             lin_solver_kwargs={"rtol": 1e-12} if solver == "bicgstab" else None)
dt = time.perf_counter() - t0
print(json.dumps({"solver": solver, "n_unknowns": n * n, "success": bool(res.success),
                  "nit": int(res.nit), "time_s": round(dt, 2),
                  "peak_rss_MB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
                  "c_min": float(res.x.min())}))
"""


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 450

    for solver in ("default", "bicgstab"):
        out = subprocess.run([sys.executable, "-c", CODE, str(n), solver],
                             capture_output=True, text=True)
        line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr[-300:]
        print(line, flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
