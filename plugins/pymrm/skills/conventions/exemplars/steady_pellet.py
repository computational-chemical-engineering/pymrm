"""Steady diffusion and reaction in a spherical catalyst pellet (structure S3).

Model (dimensionless, radius 1):

    (1/r^2) d/dr (r^2 dc/dr) = phi^2 c^m,      0 < r < 1
    dc/dr = 0 at r = 0 (symmetry),   c = 1 at r = 1 (surface)

Target quantity: the effectiveness factor

    eta = (surface flux * area) / (volume * rate at surface conditions)
        = 3 * dc/dr|_{r=1} / phi^2

Assumptions: isothermal pellet, constant effective diffusivity, no external
film resistance, power-law kinetics of order m.

Checks, each able to fail:
1. m = 1 against the closed form eta = 3 / phi^2 * (phi coth(phi) - 1).
2. Grid refinement: observed order of the eta error (expected 2).
3. m = 2 by a second, independent route: shooting on the ODE with solve_ivp.
   Shares no code with the finite-volume model.
4. Break row: the same run with nu = 0 (slab) must move eta, proving the checks
   see the geometry.
The volume-integrated rate equals the surface flux to round-off for any grid and
any nu, because the discrete divergence telescopes. That is a STRUCTURAL
identity: it cannot fail, so it is not reported as evidence.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from pymrm import NumJac, compute_boundary_values, construct_div, construct_grad, newton

try:
    from pymrm.checks import residual_check
except ImportError:  # pymrm 2.3.1 or older: the copy shipped with the plugin
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from pymrm_checks import residual_check


class SteadyPellet:
    """Finite-volume model of a pellet with power-law kinetics."""

    def __init__(self, phi=2.0, order=1.0, n_r=100, nu=2):
        self.phi = phi
        self.order = order
        self.n_r = n_r
        self.nu = nu  # 2: sphere, 1: infinite cylinder, 0: slab

        self._build_grid()
        # r = 0: dc/dr = 0 (symmetry); r = 1: c = 1 (surface concentration)
        self.bc = (
            {"a": 1.0, "b": 0.0, "d": 0.0},
            {"a": 0.0, "b": 1.0, "d": 1.0},
        )
        self.c = np.ones((self.n_r, 1))  # one field, so shape (n_r, 1)
        self._build_operators()

    def _build_grid(self):
        self.r_f = np.linspace(0.0, 1.0, self.n_r + 1)
        self.r_c = 0.5 * (self.r_f[:-1] + self.r_f[1:])

    def _build_operators(self):
        shape = self.c.shape
        grad_mat, grad_bc = construct_grad(shape, self.r_f, self.r_c, self.bc, axis=0)
        div_mat = construct_div(shape, self.r_f, nu=self.nu, axis=0)
        # residual: div(-grad c) + phi^2 c^m = 0
        self.jac_const = (div_mat @ -grad_mat).tocsc()
        self.g_const = (div_mat @ -grad_bc).toarray().reshape((-1, 1))
        self.numjac = NumJac(shape)

    def reaction(self, c):
        return self.phi**2 * np.maximum(c, 0.0) ** self.order

    def residual(self, c):
        g_rxn, jac_rxn = self.numjac(self.reaction, c.reshape(self.c.shape))
        g = self.g_const + self.jac_const @ c.reshape((-1, 1)) + g_rxn.reshape((-1, 1))
        return g, self.jac_const + jac_rxn

    def solve(self):
        result = newton(self.residual, self.c)
        check = residual_check(self.residual, result.x)   # scale-free, not an absolute threshold
        if not (result.success and check["ok"]):
            raise RuntimeError(f"pellet solve failed: {result.message}, backward error "
                               f"{check['backward_error']:.1e}")
        self.c = result.x.reshape(self.c.shape)
        return self

    def effectiveness_factor(self):
        # gradient along +r, which is the outward normal at r = 1
        _, _, _, grad_surface = compute_boundary_values(self.c, self.r_f, self.r_c, self.bc)
        return (self.nu + 1) * float(np.ravel(grad_surface)[0]) / self.phi**2


def eta_first_order_sphere(phi):
    return 3.0 / phi**2 * (phi / np.tanh(phi) - 1.0)


def eta_by_shooting(phi, order):
    """Second route: integrate from the centre, find c(0) such that c(1) = 1."""

    def rhs(r, y):
        c, dc = y
        # c'' + (2/r) c' = phi^2 c^m; near r = 0 use the limit c'' = phi^2 c^m / 3
        d2c = phi**2 * max(c, 0.0) ** order - (2.0 / r) * dc if r > 0 else phi**2 * max(c, 0.0) ** order / 3.0
        return [dc, d2c]

    def surface_mismatch(c_centre):
        sol = solve_ivp(rhs, (0.0, 1.0), [c_centre, 0.0], rtol=1e-11, atol=1e-13)
        return sol.y[0, -1] - 1.0, sol.y[1, -1]

    c_centre = brentq(lambda c0: surface_mismatch(c0)[0], 1e-6, 1.0, xtol=1e-14)
    return 3.0 * surface_mismatch(c_centre)[1] / phi**2


def run_checks(verbose=True):
    phi = 2.0
    results = {}

    exact = eta_first_order_sphere(phi)
    errors = []
    for n_r in (25, 50, 100, 200):
        eta = SteadyPellet(phi=phi, order=1.0, n_r=n_r).solve().effectiveness_factor()
        errors.append(abs(eta - exact))
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    results["first_order_rel_error_n200"] = errors[-1] / exact
    results["observed_order"] = orders[-1]

    eta_fv = SteadyPellet(phi=phi, order=2.0, n_r=400).solve().effectiveness_factor()
    eta_shoot = eta_by_shooting(phi, 2.0)
    results["second_order_two_routes_rel_diff"] = abs(eta_fv - eta_shoot) / eta_shoot

    eta_slab = SteadyPellet(phi=phi, order=1.0, n_r=200, nu=0).solve().effectiveness_factor()
    results["break_row_slab_shift"] = abs(eta_slab - exact) / exact

    passed = {
        "first_order_rel_error_n200": results["first_order_rel_error_n200"] < 1e-4,
        "observed_order": abs(results["observed_order"] - 2.0) < 0.1,
        "second_order_two_routes_rel_diff": results["second_order_two_routes_rel_diff"] < 1e-4,
        "break_row_slab_shift": results["break_row_slab_shift"] > 0.05,
    }
    if verbose:
        print(f"eta (m=1, phi={phi}) exact            = {exact:.8f}")
        print(f"eta (m=2) finite volume n=400        = {eta_fv:.8f}")
        print(f"eta (m=2) shooting                   = {eta_shoot:.8f}")
        for key, value in results.items():
            print(f"{key:36s} = {value:.3e}   {'PASS' if passed[key] else 'FAIL'}")
    return results, passed


if __name__ == "__main__":
    run_checks()
