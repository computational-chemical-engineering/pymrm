"""Ignition and extinction of a non-isothermal catalyst pellet by continuation (structure S3, compact script).

1. Problem statement
   First-order exothermic reaction in a spherical pellet (Weisz-Hicks), with the
   Prater relation T = 1 + beta (1 - y) folded into the Arrhenius rate:

       (1/u^2) d/du (u^2 dy/du) = phi^2 R(y),   R(y) = y exp(gamma beta (1 - y) / (1 + beta (1 - y)))
       dy/du = 0 at u = 0 (symmetry),  y = 1 at u = 1 (surface)

   y = c/c_s, u = r/R, phi = Thiele modulus (the swept parameter), effectiveness
   factor eta = volume average of R(y). For large enough beta and gamma, eta(phi)
   is S-shaped: for phi between the extinction and ignition values the pellet has
   a cold and a hot steady state, plus an unstable middle one that is not reachable.
   Assumptions: steady, spherical, constant effective properties, no external
   film resistance. Parameters are illustrative (label: assumed; the form follows
   Weisz and Hicks 1962).

   What pymrm.continuation does: every sweep value starts from the previous
   solution (secant predictor, step halving). A fold is a turning point of the
   curve in phi, so natural-parameter continuation cannot pass it: it stops and
   reports `success = False`, `last_param` (the last phi solved) and a message;
   the fold lies between `last_param` and the next value. The branch-jump check
   (max_change) is what keeps the corrector from silently hopping to the other
   branch. From pymrm 3.4.0 a repeated or nearly repeated sweep value (0.1 and
   1 - 0.9) is harmless; one such value is included in the up sweep below.

   Checks, each able to fail:
   - Fold check: the up sweep from the cold branch must stop (success False) at
     the ignition fold, the down sweep from the hot branch at the extinction
     fold, with phi_extinction < phi_ignition.
   - Window check: at a phi between the two stops both sweeps have solutions on
     different branches (centre temperature differs by > 0.3 in units of T_s).
   - Second route: a shooting solution from the centre (solve_ivp plus brentq on
     the centre value, no shared code) reproduces the effectiveness factor of a
     cold and a hot point within 1e-3 relative.
   - Break row: with beta = 0.1 (no multiplicity) both sweeps reach every value.
   Note: scripts/check_model.py reports one P7 line here; it is the corrector
   failing at the fold on purpose, which is how continuation detects it.
"""

# 2. Imports
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from pymrm import NumJac, construct_div, construct_grad, continuation

# 3. Parameters (all assumed, for illustration)
BETA = 0.6         # Prater number (maximum relative temperature rise)
GAMMA = 20.0       # Arrhenius number E / (R_g T_s)
N_R = 100          # cells over 0 < u < 1
PHI_UP = [0.1, 1.0 - 0.9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]   # 1 - 0.9 repeats 0.1 nearly
PHI_DOWN = [1.5, 1.2, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
PHI_WINDOW = 0.4   # inside the multiplicity window of the baseline
NEWTON = {"tol": 1e-10, "maxfev": 40}

# 4. Model equations and grid
shape = (N_R, 1)                                    # one field: (n, 1)
r_f = np.linspace(0.0, 1.0, N_R + 1)                # faces
r_c = 0.5 * (r_f[:-1] + r_f[1:])                    # centres
bc = ({"a": 1.0, "b": 0.0, "d": 0.0},               # u = 0 (outward normal -u): -dy/du = 0, symmetry
      {"a": 0.0, "b": 1.0, "d": 1.0})               # u = 1: y = 1, surface value
grad_mat, grad_bc = construct_grad(shape, r_f, r_c, bc)
div_mat = construct_div(shape, r_f, nu=2)           # nu = 2: sphere
jac_diff = (div_mat @ (-grad_mat)).tocsc()          # div(-grad) y, flux J = -dy/du
g_diff = (div_mat @ (-grad_bc)).toarray()           # constant boundary part, shape (n, 1)
weights = r_f[1:]**3 - r_f[:-1]**3                  # sphere cell volumes, sum = 1
numjac = NumJac(shape)                              # the rate is local: diagonal Jacobian only


def make_rate(beta, gamma=GAMMA):
    def rate(y):
        # 1 - y >= -0.5/beta keeps the Prater denominator positive, so a Newton overshoot cannot overflow exp
        s = np.maximum(1.0 - y, -0.5 / beta)
        return y * np.exp(gamma * beta * s / (1.0 + beta * s))
    return rate


def make_residual(beta):
    rate = make_rate(beta)

    def residual(y, phi):
        """g = -(laplacian y) + phi^2 R(y) (flux J = -dy/du), with its Jacobian."""
        g_r, jac_r = numjac(rate, y.reshape(shape))
        g = jac_diff @ y.reshape(shape) + g_diff + phi**2 * g_r   # -(lap y) + phi^2 R
        return g.ravel(), (jac_diff + phi**2 * jac_r).tocsc()
    return residual


# 5. Solver: continuation sweeps
def sweep(beta, phis, y_start):
    result = continuation(make_residual(beta), y_start, phis, newton_kwargs=NEWTON)
    if not result.success and result.last_param is None:
        raise RuntimeError(f"sweep failed at its first value: {result.message}")
    return result


def eta_of(y, beta):
    return float(weights @ make_rate(beta)(np.asarray(y).ravel()))


def centre_temperature(y, beta):
    return 1.0 + beta * (1.0 - float(np.asarray(y).ravel()[0]))   # Prater, in units of T_s


def sweeps(beta):
    """Up from the cold branch, down from a hot-branch solution found at the top value."""
    up = sweep(beta, PHI_UP, np.ones(N_R))
    hot_guess = np.full(N_R, 0.05)    # low centre concentration: the hot branch
    down = sweep(beta, PHI_DOWN, hot_guess)
    return up, down


# 6. Independent route: shooting from the centre (shares no code with the above)
def shoot(beta, y_c):
    """Integrate from the centre; with s = phi u phi drops out and s at y = 1 IS phi."""
    rate = make_rate(beta)

    def rhs(s, v):
        return [v[1], rate(v[0]) - 2.0 / s * v[1]]

    def hit(s, v):
        return v[0] - 1.0
    hit.terminal, hit.direction = True, 1
    s0 = 1e-6
    v0 = [y_c + rate(y_c) * s0**2 / 6.0, rate(y_c) * s0 / 3.0]    # series at the centre
    sol = solve_ivp(rhs, (s0, 60.0), v0, events=hit, rtol=1e-10, atol=1e-12)
    if not sol.t_events[0].size:
        return None
    s_star = float(sol.t_events[0][0])
    return s_star, 3.0 * float(sol.y_events[0][0][1]) / s_star    # (phi, eta)


def shooting_eta(beta, phi, bracket):
    """eta at Thiele modulus phi for the centre value in bracket (it selects the branch).
    The hot branch has a centre value near zero, so its bracket is on a log scale."""
    def miss(y_c):
        out = shoot(beta, y_c)
        return (out[0] if out else 60.0) - phi
    y_c = brentq(miss, *bracket, xtol=1e-14, rtol=1e-12)
    return shoot(beta, y_c)[1]


# 7. Post-processing and validation
def at(result, phi):
    return result.xs[int(np.argmin(np.abs(np.asarray(result.params) - phi)))]


def run_checks(verbose=True):
    up, down = sweeps(BETA)
    results = {"phi_ignition_last": float(up.last_param), "phi_extinction_last": float(down.last_param)}
    if verbose:
        print("up  :", up.success, "|", up.message)
        print("down:", down.success, "|", down.message)
        print(f"ignition fold between {up.last_param:.3f} and the next value; "
              f"extinction fold between {down.last_param:.3f} and the next value")
    cold, hot = at(up, PHI_WINDOW), at(down, PHI_WINDOW)
    results["window_dT_centre"] = centre_temperature(hot, BETA) - centre_temperature(cold, BETA)
    err = max(abs(shooting_eta(BETA, PHI_WINDOW, b) - eta_of(y, BETA)) / eta_of(y, BETA)
              for y, b in ((cold, (0.9, 0.9999)), (hot, (1e-9, 1e-2))))
    results["two_routes_max_rel_diff_eta"] = err
    if verbose:
        print(f"phi = {PHI_WINDOW}: eta cold {eta_of(cold, BETA):.4f}, hot {eta_of(hot, BETA):.4f}")

    up0, down0 = sweeps(0.1)             # break row: no multiplicity
    results["break_row_unreached_values"] = float(len(PHI_UP) - len(up0.params) + len(PHI_DOWN) - len(down0.params))

    passed = {
        "phi_ignition_last": (not up.success) and 0.5 < up.last_param < 0.7,
        "phi_extinction_last": (not down.success) and 0.2 < down.last_param < up.last_param,
        "window_dT_centre": results["window_dT_centre"] > 0.3,
        "two_routes_max_rel_diff_eta": err < 1e-3,
        "break_row_unreached_values": up0.success and down0.success
        and results["break_row_unreached_values"] == 0.0,
    }
    if verbose:
        for key, value in results.items():
            print(f"{key:30s} = {value:.3e}   {'PASS' if passed[key] else 'FAIL'}")
    return results, passed


if __name__ == "__main__":
    run_checks()
