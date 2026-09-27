"""Run a pymrm model script with pymrm instrumented, and report pitfall symptoms.

Usage:
    python check_model.py path/to/model.py [script arguments]

The script runs normally (its own output is shown); afterwards a report lists
what the instrumentation saw. It catches, at run time:

- P3: NumJac on a one-dimensional shape of 100 or more with the default
      stencil (dense Jacobian; a small pointwise (n_c,) system is fine);
- P4: NumJac with axes_diagonals on a one-dimensional shape, only on pymrm
      versions that still build a wrong Jacobian there (2.3.1 and older);
- P7: newton finishing with success while its step tolerance is not small
      against the size of the unknowns (absolute-tolerance trap), or a newton
      call that did not succeed;
- deprecated update_csc_array_indices calls.

A clean report is not proof of correctness; it only means these symptoms were
not seen on the code paths the run executed.
"""

import runpy
import sys

import numpy as np

import pymrm
import pymrm.numjac
import pymrm.solve

FINDINGS = []


def _note(code, message):
    if (code, message) not in FINDINGS:
        FINDINGS.append((code, message))


_numjac_init = pymrm.numjac.NumJac.__init__


def _old_1d_stencil():
    """True if this pymrm still misreads axes_diagonals on a 1-D shape (P4)."""
    deps = pymrm.numjac.stencil_block_diagonals(ndims=1, axes_diagonals=[0])
    return any(len(dep) > 2 and list(dep[2]) == [-1] for dep in deps)


OLD_1D_STENCIL = _old_1d_stencil()


def _checked_numjac_init(self, shape=None, *args, **kwargs):
    shape_t = (shape,) if isinstance(shape, (int, np.integer)) else tuple(shape) if shape is not None else None
    if shape_t is not None and len(shape_t) == 1 and shape_t[0] > 1:
        if kwargs.get("axes_diagonals"):
            if OLD_1D_STENCIL:
                _note("P4", f"NumJac({shape_t}, axes_diagonals=...) on a 1-D shape gives a WRONG Jacobian "
                            f"in this pymrm version; use shape {(shape_t[0], 1)}")
        elif shape_t[0] >= 100 and "axes_blocks" not in kwargs:
            _note("P3", f"NumJac({shape_t}) on a 1-D shape builds a dense {shape_t[0]} x {shape_t[0]} "
                        f"Jacobian; for a field on a grid use shape {(shape_t[0], 1)}")
    return _numjac_init(self, shape, *args, **kwargs)


_newton = pymrm.solve.newton


def _checked_newton(function, initial_guess, *args, **kwargs):
    result = _newton(function, initial_guess, *args, **kwargs)
    tol = kwargs.get("tol", 1.49012e-08)
    x = np.asarray(result.x, dtype=float).ravel()
    scale = np.max(np.abs(x)) if x.size else 0.0
    if not result.success:
        _note("P7", f"newton did not succeed ({result.message}); check the final residual before using x")
    elif scale > 0.0 and tol > 1e-4 * scale:
        _note("P7", f"newton stopped on an absolute step tolerance {tol:g} while max|x| = {scale:.3g}; "
                    f"scale the unknowns to order one (or lower tol) and check the final residual")
    return result


def _checked_update_csc(*args, **kwargs):
    _note("API", "update_csc_array_indices is deprecated; use update_array_indices")
    return _update_csc(*args, **kwargs)


_update_csc = pymrm.update_csc_array_indices

pymrm.NumJac.__init__ = _checked_numjac_init
pymrm.newton = _checked_newton
pymrm.solve.newton = _checked_newton
pymrm.update_csc_array_indices = _checked_update_csc


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    status = 0
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as exc:
        status = exc.code if isinstance(exc.code, int) else 0
    print("\n=== pymrm check_model report ===")
    if not FINDINGS:
        print("no pitfall symptoms seen on the executed code paths")
    for code, message in FINDINGS:
        print(f"[{code}] {message}")
    return 1 if FINDINGS else status


if __name__ == "__main__":
    sys.exit(main())
