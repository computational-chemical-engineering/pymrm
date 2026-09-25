"""Transient axially dispersed plug-flow reactor (structures S4, S5).

Model (dimensionless: z = x/L, t = time * v/L):

    dc/dt + dc/dz = (1/Pe) d2c/dz2 - Da c,       0 < z < 1
    z = 0: c - (1/Pe) dc/dz = c_in    (Danckwerts inlet)
    z = 1: dc/dz = 0                  (Danckwerts outlet)
    c(z, 0) = 0, then a step to c_in = 1 at t = 0

Target quantities: the outlet response c_out(t) and the steady outlet value.

Assumptions: constant velocity and dispersion coefficient, isothermal,
first-order reaction.

Checks, each able to fail:
1. Steady outlet against the closed-form Danckwerts solution.
2. Grid refinement of the steady outlet: observed order (upwind convection
   is first order, so expect about 1).
3. Time-step refinement of c_out at t = 0.5 during the transient: observed order
   (backward Euler, expect about 1).
4. Break row: flipping the sign of `a` in the inlet bc must move the outlet.
The integral mass balance written with face values closes to round-off for any
grid and time step: it is STRUCTURAL and is reported only as a code-consistency
check. Written with the last cell value instead of the face value it would not
close (see pitfalls P8).
"""

import numpy as np
from scipy.sparse import eye_array
from scipy.sparse.linalg import splu

from pymrm import compute_boundary_values, construct_convflux_upwind, construct_div, construct_grad


class DispersionReactor:
    """Linear model: the Jacobian is constant, so one LU factorisation serves every step."""

    def __init__(self, peclet=10.0, damkohler=1.0, n_z=200, dt=0.01, inlet_sign=1.0):
        self.peclet = peclet
        self.damkohler = damkohler
        self.n_z = n_z
        self.dt = dt
        self.c_in = 1.0

        self.z_f = np.linspace(0.0, 1.0, self.n_z + 1)
        self.z_c = 0.5 * (self.z_f[:-1] + self.z_f[1:])
        # z = 0: c - (1/Pe) dc/dz = c_in. The outward normal is -z, so
        #        -(1/Pe) dc/dz = +(1/Pe) dc/dn and a = +1/Pe.
        # z = 1: dc/dz = 0.
        self.bc = (
            {"a": inlet_sign / self.peclet, "b": 1.0, "d": self.c_in},
            {"a": 1.0, "b": 0.0, "d": 0.0},
        )
        self.c = np.zeros((self.n_z, 1))
        self.time = 0.0
        self._build_operators()

    def _build_operators(self):
        shape = self.c.shape
        grad_mat, grad_bc = construct_grad(shape, self.z_f, self.z_c, self.bc)
        conv_mat, conv_bc = construct_convflux_upwind(shape, self.z_f, self.z_c, self.bc, v=1.0)
        div_mat = construct_div(shape, self.z_f, nu=0)  # nu=0: Cartesian
        self.flux_mat = conv_mat - grad_mat / self.peclet
        self.flux_bc = (conv_bc - grad_bc / self.peclet).toarray().reshape((-1, 1))
        # steady residual: div(flux) + Da c = 0
        self.jac_steady = (div_mat @ self.flux_mat + self.damkohler * eye_array(self.n_z)).tocsc()
        self.g_const = div_mat @ self.flux_bc
        self.lu_steady = splu(self.jac_steady)
        self.lu_step = splu((self.jac_steady + eye_array(self.n_z) / self.dt).tocsc())

    def step(self):
        # backward Euler: (c - c_old)/dt + jac_steady c + g_const = 0
        rhs = self.c.reshape((-1, 1)) / self.dt - self.g_const
        self.c = self.lu_step.solve(rhs.ravel()).reshape(self.c.shape)
        self.time += self.dt

    def solve_until(self, t_end, callback=None):
        n_steps = int(round(t_end / self.dt))
        for _ in range(n_steps):
            self.step()
            if callback is not None:
                callback(self)
        return self

    def solve_steady(self):
        self.c = self.lu_steady.solve(-self.g_const.ravel()).reshape(self.c.shape)
        return self

    def outlet(self):
        _, _, c_out, _ = compute_boundary_values(self.c, self.z_f, self.z_c, self.bc)
        return float(np.ravel(c_out)[0])

    def total_flux_faces(self):
        return (self.flux_mat @ self.c.reshape((-1, 1)) + self.flux_bc).ravel()


def outlet_exact(peclet, damkohler):
    a = np.sqrt(1.0 + 4.0 * damkohler / peclet)
    num = 4.0 * a * np.exp(peclet / 2.0)
    den = (1.0 + a) ** 2 * np.exp(a * peclet / 2.0) - (1.0 - a) ** 2 * np.exp(-a * peclet / 2.0)
    return num / den


def run_checks(verbose=True):
    pe, da = 10.0, 1.0
    results = {}

    exact = outlet_exact(pe, da)
    errors = []
    for n_z in (100, 200, 400, 800):
        errors.append(abs(DispersionReactor(pe, da, n_z=n_z).solve_steady().outlet() - exact))
    results["steady_outlet_rel_error_n800"] = errors[-1] / exact
    results["grid_order"] = np.log2(errors[-2] / errors[-1])

    # time-step refinement at t = 0.5 on a fine grid; the reference is dt/8
    values = [DispersionReactor(pe, da, n_z=400, dt=dt).solve_until(0.5).outlet()
              for dt in (0.02, 0.01, 0.005, 0.0025)]
    diffs = np.abs(np.diff(values))
    results["time_order"] = np.log2(diffs[-2] / diffs[-1])

    # structural balance: in - out - reacted = accumulated, using face fluxes
    model = DispersionReactor(pe, da, n_z=200, dt=0.01)
    dz = np.diff(model.z_f)
    budget = {"net_in": 0.0}

    def accumulate(m):
        flux = m.total_flux_faces()
        budget["net_in"] += m.dt * (flux[0] - flux[-1] - da * np.sum(m.c.ravel() * dz))

    model.solve_until(1.0, callback=accumulate)
    results["structural_balance_residual"] = abs(budget["net_in"] - np.sum(model.c.ravel() * dz))

    broken = DispersionReactor(pe, da, n_z=800, inlet_sign=-1.0).solve_steady().outlet()
    results["break_row_inlet_sign_shift"] = abs(broken - exact) / exact

    passed = {
        "steady_outlet_rel_error_n800": results["steady_outlet_rel_error_n800"] < 5e-3,
        "grid_order": 0.8 < results["grid_order"] < 1.2,
        "time_order": 0.8 < results["time_order"] < 1.2,
        "structural_balance_residual": results["structural_balance_residual"] < 1e-10,
        "break_row_inlet_sign_shift": results["break_row_inlet_sign_shift"] > 0.01,
    }
    if verbose:
        print(f"steady outlet exact (Pe={pe}, Da={da}) = {exact:.8f}")
        for key, value in results.items():
            print(f"{key:36s} = {value:.3e}   {'PASS' if passed[key] else 'FAIL'}")
    return results, passed


if __name__ == "__main__":
    run_checks()
