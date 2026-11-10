"""Crossings, thresholds and extrema of correlations (pointwise algebra, compact script format).

1. Problem statement
   Two correlations, no finite-volume model; every number is a root or a limit.

   (a) Wen & Yu (1966) minimum fluidisation, the positive root of a quadratic in Re = Re_mf:

           Re^2 + 2 C1 Re - C2 Ga = 0,    Re = sqrt(C1^2 + C2 Ga) - C1

   (b) Reilly et al. (1994) type regime transition of a bubble column: with
       eps(rho) = A rho^s the transition holdup and V(rho) = V0 rho^q the small-bubble velocity,

           U(rho) = V(rho) eps (1 - eps),     eps -> 1 is a pole of d ln U / d ln rho.

   Targets: the Re that meets a given Ga (a crossing), the density rho* at which U is largest
   (an extremum) and the limits Ga -> 0, Ga -> infinity and rho -> 0 in closed form.
   Rules: a crossing is found by pymrm.checks.find_roots on a physically possible bracket and
   must be unique, or the bracket or the model is wrong; an extremum is the root of the ANALYTIC
   derivative with the pole multiplied out, never the maximum over a grid; a limit is a formula,
   never the value at the end of a grid.

   Parameters (label: source)
   C1 = 33.7, C2 = 0.0408          Wen & Yu (1966), eq. (1), as printed
   s = 0.48, q = -0.04             Reilly et al. (1994), eq. (8), density exponents (rho^0.96 under the square root)
   A = 0.1 (kg/m3)^-s, V0 = 0.1 m/s    assumed, for illustration (the paper's constants depend on the liquid)

   Checks, each able to fail:
   - find_roots on the quadratic agrees with the closed-form root to 1e-9 relative.
   - The exactly-one-root helper raises on a bracket with two roots and on one with none.
   - The root of the analytic derivative matches a 20001-point grid maximum to one grid step in
     ln rho, and the derivative changes sign from + to - there.
   - Each closed-form limit matches the full correlation far into its regime (1e-4 relative).
   - Break row: C1 raised by 20 percent must move the Re root by more than 5 percent.
"""

# 2. Imports
import numpy as np

try:
    from pymrm.checks import find_roots
except ImportError:  # pymrm 2.3.1 or older: the copy shipped with the plugin
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from pymrm_checks import find_roots

# 3. Parameters
C1, C2 = 33.7, 0.0408        # Wen & Yu (1966) eq. (1)
S, Q = 0.48, -0.04           # Reilly et al. (1994) eq. (8) exponents
A, V0 = 0.1, 0.1             # assumed prefactors
GA = 1.0e5                   # assumed Galileo number for the crossing
RHO_POLE = (1.0 / A) ** (1.0 / S)   # eps = 1: the correlation is physical only below it
RHO_LO = 0.1                 # kg/m3, assumed lowest density of interest


# 4. Model equations
def quadratic(re, ga=GA, c1=C1):
    """Residual of the Wen-Yu force balance, the function whose root is Re."""
    return re**2 + 2.0 * c1 * re - C2 * ga


def re_closed(ga, c1=C1):
    return np.sqrt(c1**2 + C2 * np.asarray(ga, float)) - c1


def holdup(rho):
    return A * np.asarray(rho, float) ** S


def u_trans(rho):
    eps = holdup(rho)
    return V0 * np.asarray(rho, float) ** Q * eps * (1.0 - eps)


def dlnu_times_one_minus_eps(rho):
    """(1 - eps) d ln U / d ln rho = q (1 - eps) + s (1 - 2 eps): the pole at eps = 1 is gone."""
    eps = holdup(rho)
    return Q * (1.0 - eps) + S * (1.0 - 2.0 * eps)


# 5. Solver
def unique_root(f, lo, hi, what, **kw):
    """The single root of f on [lo, hi]. Raises unless find_roots finds exactly one, so a second
    crossing or an empty bracket is never silently replaced by 'the first root'."""
    found = find_roots(f, lo, hi, **kw)
    if len(found["roots"]) != 1:
        raise ValueError(f"{what}: expected one root on [{lo:g}, {hi:g}], got {found['message']}")
    return found["roots"][0]


def raises(f, lo, hi, **kw):
    try:
        unique_root(f, lo, hi, "test", **kw)
    except ValueError:
        return True
    return False


# 6. Post-processing and 7. validation
def run_checks(verbose=True):
    results, passed = {}, {}

    # (1) crossing: find_roots on a positive log bracket against the closed form
    re_num = unique_root(quadratic, 1e-6, 1e4, "Re at given Ga", log=True)
    results["root_rel_diff"] = abs(re_num - float(re_closed(GA))) / float(re_closed(GA))
    passed["root_rel_diff"] = results["root_rel_diff"] < 1e-9

    # (2) the helper raises: [-1e3, 1e3] holds two roots (Re > 0 and its negative partner), [1e-6, 1e-3] none
    results["raises_two_roots"] = float(raises(quadratic, -1e3, 1e3))
    results["raises_no_root"] = float(raises(quadratic, 1e-6, 1e-3, log=True))
    passed["raises_two_roots"] = results["raises_two_roots"] == 1.0
    passed["raises_no_root"] = results["raises_no_root"] == 1.0

    # (3) extremum: root of the analytic derivative, closed form eps* = (q + s)/(q + 2 s)
    rho_star = unique_root(dlnu_times_one_minus_eps, RHO_LO, RHO_POLE, "peak of U_trans", log=True)
    grid = np.geomspace(RHO_LO, RHO_POLE * (1.0 - 1e-9), 20001)
    step = np.log(grid[1] / grid[0])
    rho_grid = grid[np.argmax(u_trans(grid))]
    results["peak_rho"] = rho_star
    results["peak_rho_vs_grid_in_steps"] = abs(np.log(rho_star / rho_grid)) / step
    results["peak_eps_vs_closed_form"] = abs(float(holdup(rho_star)) - (Q + S) / (Q + 2.0 * S))
    results["derivative_before_after"] = float(
        dlnu_times_one_minus_eps(0.5 * rho_star) > 0.0 > dlnu_times_one_minus_eps(2.0 * rho_star))
    passed["peak_rho"] = RHO_LO < rho_star < RHO_POLE
    passed["peak_rho_vs_grid_in_steps"] = results["peak_rho_vs_grid_in_steps"] <= 1.0
    passed["peak_eps_vs_closed_form"] = results["peak_eps_vs_closed_form"] < 1e-9
    passed["derivative_before_after"] = results["derivative_before_after"] == 1.0

    # (4) closed-form limits against the full correlation deep in each regime
    low = float(re_closed(1e-6)) / 1e-6 / (C2 / (2.0 * C1))              # Re/Ga -> C2/(2 C1)
    high = float(re_closed(1e14)) / np.sqrt(1e14) / np.sqrt(C2)          # Re/sqrt(Ga) -> sqrt(C2)
    thin = float(u_trans(1e-8)) / (V0 * A * 1e-8 ** (Q + S))             # U -> V0 A rho^(q+s) as rho -> 0
    for key, ratio in (("limit_low_Ga", low), ("limit_high_Ga", high), ("limit_low_rho", thin)):
        results[key] = abs(ratio - 1.0)
        passed[key] = results[key] < 1e-4

    # (5) break row: C1 up by 20 percent must move the root
    re_broken = unique_root(lambda r: quadratic(r, c1=1.2 * C1), 1e-6, 1e4, "broken Re", log=True)
    results["break_row_C1_shift"] = abs(re_broken - re_num) / re_num
    passed["break_row_C1_shift"] = results["break_row_C1_shift"] > 0.05

    if verbose:
        for key, value in results.items():
            print(f"{key:28s} = {value:.3e}   {'PASS' if passed[key] else 'FAIL'}")
    return results, passed


if __name__ == "__main__":
    run_checks()
