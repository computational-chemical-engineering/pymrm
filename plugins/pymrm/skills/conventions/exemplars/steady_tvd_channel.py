"""Steady 2-D channel with a catalytic wall, TVD convection solved by Newton (structure S5).

Model (dimensionless: x and y scaled by the channel half-height H, velocity by u_max):

    u(y) dc/dx = D d2c/dy2,                 0 < x < L, 0 < y < 1
    u(y) = 2 y - y**2                       (0 at the wall, 1 at the centre line)
    x = 0: c = 1                            (inlet)
    x = L: outflow                          (pure convection along x)
    y = 0: D dc/dy = k_w c                  (first-order wall reaction)
    y = 1: dc/dy = 0                        (symmetry)

Target quantity: the conversion 1 - (outlet flow) / (inlet flow).

Assumptions: fully developed laminar flow, no axial diffusion, isothermal,
first-order wall kinetics.

Convection along x uses a van Leer limiter through `ConvFluxTVD`, which returns
the face fluxes and their exact Jacobian; Newton with `damping="monotone"` then
converges in a handful of steps. Deferred correction (the limiter correction in
the residual with the upwind Jacobian) does not converge for this steady problem
(pitfalls P10).

Checks, each able to fail:
1. Newton converges, and the Jacobian matches a directional finite difference.
2. Refinement along x (n_y fixed): observed order of the conversion, about 1 for
   first-order upwind and close to 2 for van Leer.
3. The concentration stays within [0, 1].
4. Break row: the same model with the upwind Jacobian (deferred correction
   inside undamped Newton) does not converge. With `damping="monotone"` it does
   for van Leer after about 30 iterations instead of 9, and for minmod only
   after about 220.
The mass balance inlet - outlet = wall uptake, written with the face fluxes,
closes to round-off on any grid: it is STRUCTURAL and is reported only as a
code-consistency check.
"""

import numpy as np

from pymrm import ConvFluxTVD, construct_convflux_upwind, construct_div, construct_grad, newton, upwind, vanleer


class Channel:
    def __init__(self, n_x=80, n_y=20, length=4.0, diffusivity=0.1, k_wall=0.5, limiter=vanleer):
        self.shape = (n_x, n_y)
        self.x_f = np.linspace(0.0, length, n_x + 1)
        self.y_f = np.linspace(0.0, 1.0, n_y + 1)
        y_c = 0.5 * (self.y_f[:-1] + self.y_f[1:])
        self.u = 2 * y_c - y_c**2
        self.v = np.broadcast_to(self.u[None, :], (n_x + 1, n_y))     # face velocity along x
        self.diffusivity, self.k_wall = diffusivity, k_wall

        # x = 0: c = 1; x = L: outflow
        self.bc_x = ({"a": 0.0, "b": 1.0, "d": 1.0}, {"outflow": True})
        # y = 0, outward normal -y: D dc/dy = k_w c  <=>  D dc/dn + k_w c = 0
        # y = 1: dc/dn = 0
        self.bc_y = ({"a": diffusivity, "b": k_wall, "d": 0.0}, {"a": 1.0, "b": 0.0, "d": 0.0})
        self.tvd = ConvFluxTVD(self.shape, self.x_f, bc=self.bc_x, tvd_limiter=limiter, axis=0)
        self.div_x = construct_div(self.shape, self.x_f, nu=0, axis=0)
        grad_y, grad_y_bc = construct_grad(self.shape, self.y_f, bc=self.bc_y, axis=1)
        div_y = construct_div(self.shape, self.y_f, nu=0, axis=1)
        self.grad_y, self.grad_y_bc = grad_y, grad_y_bc.toarray().ravel()
        self.diff = (-diffusivity * (div_y @ grad_y)).tocsc()           # constant: built once
        self.diff_bc = -diffusivity * (div_y @ grad_y_bc).toarray().ravel()
        conv, _ = construct_convflux_upwind(self.shape, self.x_f, bc=self.bc_x, v=self.v, axis=0)
        self.jac_upwind = (self.div_x @ conv).tocsc()

    def residual(self, c):
        flux, jac = self.tvd(c, self.v)
        g = self.div_x @ flux.ravel() + self.diff @ c.ravel() + self.diff_bc
        return g, (self.div_x @ jac + self.diff).tocsc()

    def residual_upwind_jacobian(self, c):
        g, _ = self.residual(c)
        return g, (self.jac_upwind + self.diff).tocsc()

    def solve(self, upwind_jacobian=False, maxfev=50, damping="monotone"):
        fun = self.residual_upwind_jacobian if upwind_jacobian else self.residual
        self.result = newton(fun, np.ones(self.shape), tol=1e-12, maxfev=maxfev, damping=damping)
        self.c = self.result.x.reshape(self.shape)
        return self

    def conversion(self):
        flux, _ = self.tvd(self.c, self.v)                             # face fluxes (pitfalls P8)
        dy = np.diff(self.y_f)
        return 1.0 - np.sum(flux[-1] * dy) / np.sum(flux[0] * dy)

    def balance_residual(self):
        flux, _ = self.tvd(self.c, self.v)
        dy, dx = np.diff(self.y_f), np.diff(self.x_f)
        wall_flux = -self.diffusivity * (self.grad_y @ self.c.ravel() + self.grad_y_bc).reshape(
            self.shape[0], self.shape[1] + 1)[:, 0]                    # -D dc/dy at y = 0
        uptake = -np.sum(wall_flux * dx)
        return abs(np.sum(flux[0] * dy) - np.sum(flux[-1] * dy) - uptake)


def run_checks(verbose=True):
    results = {}

    model = Channel().solve()
    results["newton_iterations"] = model.result.nit
    rng = np.random.default_rng(0)
    c0, dc, h = model.c + 0.01 * rng.standard_normal(model.shape), rng.standard_normal(model.shape), 1e-7
    _, jac = model.residual(c0)
    fd = (model.residual(c0 + h * dc)[0] - model.residual(c0 - h * dc)[0]) / (2 * h)
    results["jacobian_rel_error"] = np.abs(jac @ dc.ravel() - fd).max() / np.abs(fd).max()
    results["c_min"], results["c_max"] = model.c.min(), model.c.max()
    results["structural_balance_residual"] = model.balance_residual()

    for name, limiter in (("upwind", upwind), ("vanleer", vanleer)):
        x = [Channel(n_x=n_x, limiter=limiter).solve().conversion() for n_x in (40, 80, 160)]
        results[f"x_order_{name}"] = np.log2(abs(x[0] - x[1]) / abs(x[1] - x[2]))
    results["conversion"] = model.conversion()

    broken = Channel().solve(upwind_jacobian=True, maxfev=50, damping="none")
    results["break_row_upwind_jacobian_residual"] = np.abs(broken.residual(broken.c)[0]).max()

    passed = {
        "newton_iterations": model.result.status == 0 and results["newton_iterations"] <= 15,
        "jacobian_rel_error": results["jacobian_rel_error"] < 1e-6,
        "c_min": results["c_min"] > -1e-12,
        "c_max": results["c_max"] < 1 + 1e-12,
        "structural_balance_residual": results["structural_balance_residual"] < 1e-10,
        "x_order_upwind": 0.8 < results["x_order_upwind"] < 1.2,
        "x_order_vanleer": results["x_order_vanleer"] > 1.6,
        "conversion": 0.0 < results["conversion"] < 1.0,
        "break_row_upwind_jacobian_residual": results["break_row_upwind_jacobian_residual"] > 1e-8,
    }
    if verbose:
        for key, value in results.items():
            print(f"{key:36s} = {value:.4g}   {'PASS' if passed[key] else 'FAIL'}")
    return results, passed


if __name__ == "__main__":
    run_checks()
