"""Orders of convergence, with time error split from space error (compact script format).

1. Problem statement
   Diffusion into a sphere after a step in surface concentration, dimensionless:

       d c/d tau = (1/r^2) d/dr (r^2 dc/dr),   0 < r < 1
       dc/dr = 0 at r = 0 (symmetry),  c = 1 at r = 1,  c = 0 at tau = 0

   Quantity of interest: the volume-mean uptake cbar(tau), weights = exact cell
   volumes r_f^3 differences (they sum to one), read at tau = 0.01, 0.03, 0.1.
   Assumptions: constant diffusivity, uniform grid, field shape (n, 1).
   Parameters: the grids and step counts below are assumed (label: assumed).

   Why two studies. One refinement of both space and time mixes two errors. So:
   - Grid order, time removed: the SAME discrete operator solved exact in time,
     c(t) = c_ss + V exp(Lambda t) V^-1 (c0 - c_ss), eigendecomposition of the
     constant operator. Orders from successive differences, no reference needed.
   - Time order, space removed: backward Euler on ONE fixed grid against the
     exact-in-time solution on THE SAME grid, so the shared spatial error cancels.
   - Round-off companion: an observed order means something only if the finest
     successive difference is far above the round-off floor of the route,
     cond(V) eps for the eigen route.

   Reference (used for check 4 and the split only): the Crank series
   1 - (6/pi^2) sum_k exp(-k^2 pi^2 tau)/k^2. Terms t_k decrease with a ratio
   t_(k+1)/t_k that also decreases, so the tail after N terms is at most
   t_(N+1)/(1 - t_(N+2)/t_(N+1)); N grows until that bound is below 1e-15 of the
   sum. At tau >= 0.01 this needs about 20 terms, so no short-time expansion is used.

   Checks, each able to fail:
   - Grid orders within 0.2 of 2 at every time (nested grids 25 to 400).
   - Round-off: smallest finest successive difference / (cond(V) eps) above 1e3.
   - Time orders within 0.1 of 1 at every time (steps per 0.01 of tau: 4 to 32), and the
     finest-step time error within 10 % of its Richardson estimate (a wrong limit passes the order).
   - Exact-in-time mean against the Crank series on the finest grid, 1e-5.
   - Error split at tau = 0.01 on the coarsest step: total = time part + space part, and the
     time part must be above 90 % of the total (a finer step or a coarser grid fails it); the
     space part from the series must match the one from grid extrapolation within 10 %.
   - Break row: the cylinder divergence (nu = 1) on the sphere grid with sphere
     weights must fail the series agreement; midpoint weights 3 r_c^2 dr must miss
     a sum of one by more than 1e-6 while the exact volumes stay within 1e-14.
"""

import numpy as np
from scipy.linalg import eig, solve
from scipy.sparse import eye_array
from scipy.sparse.linalg import splu

from pymrm import construct_div, construct_grad
from pymrm.checks import observed_orders

# 3. Parameters (all assumed, for illustration)
TAUS = np.array([0.01, 0.03, 0.1])   # times of the uptake readings
NS = (25, 50, 100, 200, 400)         # nested grids for the grid order
STEPS = (4, 8, 16, 32)               # backward-Euler steps per 0.01 of tau
N_FIXED = 100                        # grid of the time study and the split
TOL_SERIES = 1e-5                    # exact-in-time against the series, finest grid


# 4. Reference
def crank_uptake(tau, rtol=1e-15):
    """Mean uptake from the Crank series, summed until the measured tail bound is below rtol."""
    def terms(k):
        return np.exp(-k**2 * np.pi**2 * tau) / k**2
    n = 1
    while True:
        a, b = terms(n + 1), terms(n + 2)
        total = sum(terms(k) for k in range(1, n + 1))
        if a / (1.0 - b / a) < rtol * total:
            break
        n += 1
    return 1.0 - 6.0 / np.pi**2 * total


# 5. Model: operators and the two solvers
def build(n, nu=2, nu_div=None, weights="volume"):
    """Operator J (dc/dtau = -J c + g), source g and cell weights w on n cells."""
    r_f = np.linspace(0.0, 1.0, n + 1)
    r_c = 0.5 * (r_f[:-1] + r_f[1:])
    # outward normal n, a dc/dn + b c = d:
    #   r = 0 (n = -r): dc/dr = 0        -> a = 1, b = 0, d = 0
    #   r = 1 (n = +r): c = 1            -> a = 0, b = 1, d = 1
    bc = ({"a": 1.0, "b": 0.0, "d": 0.0}, {"a": 0.0, "b": 1.0, "d": 1.0})
    grad, grad_bc = construct_grad((n, 1), r_f, r_c, bc)
    div = construct_div((n, 1), r_f, nu=nu if nu_div is None else nu_div)
    jac = (-(div @ grad)).tocsc()
    g = (div @ grad_bc).toarray().ravel()
    w = np.diff(r_f**(nu + 1)) if weights == "volume" else (nu + 1) * r_c**nu * np.diff(r_f)
    return jac, g, w


def exact_in_time(jac, g, w, taus):
    """Means at taus of the semi-discrete solution; also cond(V) of the eigenbasis."""
    lam, vec = eig(jac.toarray())
    lam = lam.real
    c_ss = np.linalg.solve(jac.toarray(), g)
    coef = solve(vec, -c_ss)                      # c0 = 0, so c0 - c_ss = -c_ss
    means = [w @ (c_ss + (vec * np.exp(-lam * t)) @ coef).real for t in taus]
    return np.array(means), np.linalg.cond(vec)


def backward_euler(jac, g, w, taus, steps_per_unit):
    """Means at taus from backward Euler with step 0.01/steps_per_unit (taus are whole steps)."""
    dt = 0.01 / steps_per_unit
    lu = splu((eye_array(len(g), format="csc") / dt + jac).tocsc())
    c, means, k = np.zeros(len(g)), [], 0
    for target in np.rint(taus / dt).astype(int):
        while k < target:
            c = lu.solve(c / dt + g)
            k += 1
        means.append(w @ c)
    return np.array(means)


# 6. Post-processing and 7. validation
def run_checks(verbose=True):
    series = np.array([crank_uptake(t) for t in TAUS])
    results, passed = {}, {}

    # grid order, time removed (cache: one dense eig per grid)
    exact = {n: exact_in_time(*build(n), TAUS) for n in NS}
    cond_v = exact[NS[-1]][1]
    for i, tau in enumerate(TAUS):
        study = observed_orders(lambda n, i=i: exact[n][0][i], NS)
        worst = max(study["orders"][-2:], key=lambda p: abs(p - 2.0))   # finest two triples
        results[f"grid_order_tau{tau:g}"] = worst
        passed[f"grid_order_tau{tau:g}"] = abs(worst - 2.0) < 0.2
        if i == 0:
            extrapolated = study["extrapolated"]
    diff = np.min(np.abs(exact[NS[-1]][0] - exact[NS[-2]][0]))   # the smallest one decides
    results["roundoff_margin"] = diff / (cond_v * np.finfo(float).eps)
    passed["roundoff_margin"] = results["roundoff_margin"] > 1e3

    # time order, space removed: same grid, so the spatial error cancels
    jac, g, w = build(N_FIXED)
    exact_fixed = exact_in_time(jac, g, w, TAUS)[0]
    be = {m: backward_euler(jac, g, w, TAUS, m) for m in STEPS}
    gap = 0.0
    for i, tau in enumerate(TAUS):
        study = observed_orders(lambda m, i=i: be[m][i] - exact_fixed[i], STEPS)
        results[f"time_order_tau{tau:g}"] = study["orders"][-1]
        passed[f"time_order_tau{tau:g}"] = abs(study["orders"][-1] - 1.0) < 0.1
        # successive differences cannot see a wrong limit: the error itself must go to zero,
        # i.e. match its own Richardson estimate
        gap = max(gap, abs(abs(be[STEPS[-1]][i] - exact_fixed[i]) - study["error_estimate"])
                  / abs(be[STEPS[-1]][i] - exact_fixed[i]))
    results["time_error_vs_estimate"] = gap
    passed["time_error_vs_estimate"] = gap < 0.1

    # exact-in-time against the series, finest grid
    results["series_vs_exact_in_time"] = np.max(np.abs(exact[NS[-1]][0] - series))
    passed["series_vs_exact_in_time"] = results["series_vs_exact_in_time"] < TOL_SERIES

    # split at the shortest time: total = time part + space part
    total = be[STEPS[0]][0] - series[0]
    time_part = be[STEPS[0]][0] - exact_fixed[0]
    space_part = exact_fixed[0] - series[0]
    results["time_share_at_tau0.01"] = abs(time_part) / (abs(total))
    # the series-based space part must match the one from grid extrapolation (no series)
    space_extrap = exact_fixed[0] - extrapolated
    results["space_part_rel_diff"] = abs(space_part - space_extrap) / abs(space_part)
    passed["time_share_at_tau0.01"] = results["time_share_at_tau0.01"] > 0.9   # the step dominates there
    passed["space_part_rel_diff"] = results["space_part_rel_diff"] < 0.1

    # break row: cylinder divergence (nu = 1) on the sphere grid, sphere weights
    bad = exact_in_time(*build(NS[-1], nu_div=1), TAUS)[0]
    results["break_cylinder_div_error"] = np.max(np.abs(bad - series))
    passed["break_cylinder_div_error"] = results["break_cylinder_div_error"] > 100 * TOL_SERIES
    r_f = np.linspace(0.0, 1.0, NS[-1] + 1)
    mid = build(NS[-1], weights="midpoint")[2].sum()
    results["break_midpoint_sum_gap"] = abs(mid - 1.0)
    results["volume_sum_gap"] = abs(np.diff(r_f**3).sum() - 1.0)
    passed["break_midpoint_sum_gap"] = results["break_midpoint_sum_gap"] > 1e-6
    passed["volume_sum_gap"] = results["volume_sum_gap"] < 1e-14

    if verbose:
        for key, value in results.items():
            print(f"{key:28s} = {value:11.4e}   {'PASS' if passed[key] else 'FAIL'}")
    return results, passed


if __name__ == "__main__":
    run_checks()
