"""Steady states of a cooled, exothermic CSTR (structure S1, compact script format).

1. Problem statement
   First-order reaction A -> B in a cooled continuous stirred tank:

       0 = (c_in - c)/tau - k(T) c
       0 = (T_in - T)/tau + beta k(T) c - kappa (T - T_c)
       k(T) = k_ref exp(-E_R (1/T - 1/T_ref))

   Target: every steady state in the physical range, with its conversion.
   Assumptions: ideal mixing, constant density and heat capacity, constant
   coolant temperature. Parameters below are illustrative (label: assumed).

   Checks, each able to fail:
   - Route 1: reduce to one equation in T, bracket every sign change on a fine
     grid, refine each root with brentq.
   - Route 2: pymrm newton on the full two-unknown system with SCALED unknowns
     (c/c_in and T/T_ref, see pitfalls P7), started near each route-1 root.
     The two routes share no residual code.
   - Break row: with beta = 0 (no heat of reaction) exactly one steady state
     must remain.
"""

# 2. Imports
import numpy as np
from scipy.optimize import brentq
from scipy.sparse import csc_array

from pymrm import newton

try:
    from pymrm.checks import residual_check
except ImportError:  # pymrm 2.3.1 or older: the copy shipped with the plugin
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from pymrm_checks import residual_check

# 3. Parameters (all assumed, for illustration)
C_IN = 1000.0      # mol/m3, feed concentration
T_IN = 300.0       # K, feed temperature
T_C = 300.0        # K, coolant temperature
TAU = 100.0        # s, residence time
BETA = 0.2         # K m3/mol, (-dH)/(rho cp): adiabatic rise 200 K
KAPPA = 0.01       # 1/s, UA/(rho cp V)
E_R = 12000.0      # K, activation temperature
T_REF = 350.0      # K
K_REF = 1.0e-2     # 1/s, k at T_REF


# 4. Model equations
def rate_constant(temp):
    return K_REF * np.exp(-E_R * (1.0 / temp - 1.0 / T_REF))


def heat_balance_reduced(temp, beta=BETA):
    """Route 1: the mass balance solved for c, substituted in the heat balance."""
    conc = C_IN / (1.0 + TAU * rate_constant(temp))
    return (T_IN - temp) / TAU + beta * rate_constant(temp) * conc - KAPPA * (temp - T_C)


def residual_scaled(y, beta=BETA):
    """Route 2: both balances in unknowns y = [c/C_IN, T/T_REF], each row divided
    by a typical magnitude so that residuals and steps are of order one."""
    conc, temp = y[0] * C_IN, y[1] * T_REF
    k = rate_constant(temp)
    dk_dtemp = k * E_R / temp**2
    g = np.array([
        ((C_IN - conc) / TAU - k * conc) * TAU / C_IN,
        ((T_IN - temp) / TAU + beta * k * conc - KAPPA * (temp - T_C)) * TAU / T_REF,
    ])
    jac = np.array([
        [-1.0 - TAU * k, -TAU * dk_dtemp * conc * T_REF / C_IN],
        [TAU * beta * k * C_IN / T_REF,
         -1.0 + TAU * beta * dk_dtemp * conc - TAU * KAPPA],
    ])
    return g, csc_array(jac)


# 5. Solver
def roots_by_bracketing(beta=BETA):
    temps = np.linspace(280.0, 600.0, 32001)
    values = heat_balance_reduced(temps, beta)
    # a root that lands exactly on a grid point would otherwise be counted twice
    exact = list(temps[values == 0.0])
    brackets = np.nonzero(values[:-1] * values[1:] < 0.0)[0]
    return sorted(exact + [brentq(heat_balance_reduced, temps[i], temps[i + 1], args=(beta,), xtol=1e-12)
                           for i in brackets])


def root_by_newton(temp_guess, beta=BETA):
    conc_guess = C_IN / (1.0 + TAU * rate_constant(temp_guess))
    y0 = np.array([conc_guess / C_IN, temp_guess / T_REF])
    result = newton(lambda y: residual_scaled(y, beta), y0)
    check = residual_check(lambda y: residual_scaled(y, beta), result.x)
    if not (result.success and check["ok"]):
        raise RuntimeError(f"newton failed near T = {temp_guess} K: {result.message}, "
                           f"backward error {check['backward_error']:.1e}")
    return result.x[0] * C_IN, result.x[1] * T_REF


# 6. Post-processing and 7. validation
def run_checks(verbose=True):
    roots = roots_by_bracketing()
    results = {"n_steady_states": float(len(roots))}
    worst = 0.0
    for temp in roots:
        # start 2 K off the route-1 root so newton has to work
        conc_n, temp_n = root_by_newton(temp + 2.0 * np.sign(temp - T_REF + 1e-9))
        worst = max(worst, abs(temp_n - temp) / temp)
        if verbose:
            slope = (heat_balance_reduced(temp + 1e-4) - heat_balance_reduced(temp - 1e-4)) / 2e-4
            print(f"T = {temp:9.4f} K   conversion = {1.0 - conc_n / C_IN:.5f}   "
                  f"{'unstable (dF/dT > 0)' if slope > 0 else 'dF/dT < 0'}")
    results["two_routes_max_rel_diff_T"] = worst
    results["break_row_no_heat_n_states"] = float(len(roots_by_bracketing(beta=0.0)))

    passed = {
        "n_steady_states": results["n_steady_states"] == 3,
        "two_routes_max_rel_diff_T": worst < 1e-8,
        "break_row_no_heat_n_states": results["break_row_no_heat_n_states"] == 1,
    }
    if verbose:
        for key, value in results.items():
            print(f"{key:36s} = {value:.3e}   {'PASS' if passed[key] else 'FAIL'}")
    return results, passed


if __name__ == "__main__":
    run_checks()
