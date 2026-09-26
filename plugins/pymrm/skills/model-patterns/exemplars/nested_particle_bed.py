"""Packed bed with resolved catalyst particles: nested scales by Schur complement (S8).

Dimensionless. Bulk, plug flow along 0 < z < 1:
    dc_b/dz = -alpha * 3 * dc_p/dr |_{r=1}            c_b(0) = 1
Particle at every z, sphere 0 < r < 1:
    (1/r^2) d/dr (r^2 dc_p/dr) = phi^2 * f(c_p)
    dc_p/dr = 0 at r = 0;   dc_p/dr = Bi (c_b - c_p) at r = 1   (film)
f(c) = c (linear, exact solution available) or c^2 (nonlinear).

Pattern: the particle's surface condition depends on the local bulk value, so
construct_grad is called with shapes_d: the returned grad_bc_right maps the bulk
vector c_b onto the particle surface faces. With shapes_d the dictionary's d is
a COEFFICIENT on that external vector (d = Bi here), not a value.

The coupled Newton system has blocks [J_bb J_bp; J_pb J_pp]. J_pp is block
diagonal (one particle per bulk cell), so the particle unknowns are eliminated
by a Schur complement in the linear solve:
    S = J_bb - J_bp J_pp^-1 J_pb,   S dx_b = g_b - J_bp J_pp^-1 g_p,
    dx_p = J_pp^-1 (g_p - J_pb dx_b)
passed to pymrm.newton as a custom linear solver.

Checks, each able to fail:
1. Linear kinetics: outlet c_b against the exact exp(-alpha * kappa), with
   kappa = 3 Bi (phi coth phi - 1) / (phi coth phi - 1 + Bi) (film in series with
   the particle).
2. Grid refinement in z (upwind, order about 1) at fixed fine radial grid.
3. Nonlinear kinetics: the Schur-complement solve against the full monolithic
   sparse solve of the same Newton system. Same equations and operators, so
   this is a consistency check of the elimination, not an independent route.
4. Break row: Bi -> 1e6 (no film) must move the outlet by more than 5 %.
"""

import numpy as np
from scipy.sparse import bmat, csc_array, diags_array
from scipy.sparse.linalg import spsolve, splu

from pymrm import NumJac, compute_boundary_values, construct_convflux_upwind, construct_div, construct_grad, newton


class ParticleBed:
    def __init__(self, n_z=100, n_r=30, alpha=0.5, phi=2.0, biot=5.0, order=1, schur=True, nu=2):
        self.n_z, self.n_r = n_z, n_r
        self.alpha, self.phi2, self.biot, self.order, self.schur = alpha, phi**2, biot, order, schur
        z_f = np.linspace(0.0, 1.0, n_z + 1)
        r_f = np.linspace(0.0, 1.0, n_r + 1)
        self.shape_b = (n_z, 1)
        self.shape_p = (n_z, n_r, 1)            # (bulk cell, radial cell, field)

        # bulk: upwind plug flow, c_b(0) = 1, zero-gradient outlet
        bc_b = ({"a": 0.0, "b": 1.0, "d": 1.0}, {"a": 1.0, "b": 0.0, "d": 0.0})
        self.z_f, self.bc_b = z_f, bc_b
        conv, conv_bc = construct_convflux_upwind(self.shape_b, z_f, bc=bc_b, v=1.0)
        div_z = construct_div(self.shape_b, z_f, nu=0)
        self.jac_bb_transport = (div_z @ conv).tocsc()
        self.g_b_const = (div_z @ conv_bc).toarray().ravel()

        # particle: symmetry at r = 0; film at r = 1: dc/dr + Bi c = Bi c_b
        # (outward normal +r, so a = 1). d = Bi is the coefficient on c_b (shapes_d).
        bc_p = ({"a": 1.0, "b": 0.0, "d": 0.0}, {"a": 1.0, "b": biot, "d": biot})
        # grad_bc_left is discarded: the symmetry condition at r = 0 has d = 0
        grad, _, grad_bc_right = construct_grad(self.shape_p, r_f, bc=bc_p, axis=1,
                                                shapes_d=(None, (n_z, 1, 1)))
        div_r = construct_div(self.shape_p, r_f, nu=nu, axis=1)   # nu=2: sphere, 1: cylinder
        surface = nu + 1.0   # particle surface over volume, times the radius (3 for a sphere)
        self.jac_pp_transport = (div_r @ -grad).tocsc()
        self.jac_pb = (div_r @ -grad_bc_right).tocsc()          # particle rows, bulk columns
        # surface gradient: outer face (r = 1) of every particle
        outer = np.arange(n_z) * (n_r + 1) + n_r
        select = csc_array((np.ones(n_z), (np.arange(n_z), outer)), shape=(n_z, n_z * (n_r + 1)))
        self.jac_bp = (surface * alpha * select @ grad).tocsc()      # bulk rows, particle columns
        self.jac_bb = (self.jac_bb_transport + surface * alpha * select @ grad_bc_right).tocsc()
        self.numjac = NumJac(self.shape_p)
        self.n_b = n_z

    def reaction(self, c):
        return self.phi2 * c**self.order

    def residual(self, x):
        c_b, c_p = x[:self.n_b], x[self.n_b:]
        g_rxn, j_rxn = self.numjac(self.reaction, c_p.reshape(self.shape_p))
        g_p = self.jac_pp_transport @ c_p + self.jac_pb @ c_b + g_rxn.ravel()
        g_b = self.g_b_const + self.jac_bb @ c_b + self.jac_bp @ c_p
        blocks = {"bb": self.jac_bb, "bp": self.jac_bp, "pb": self.jac_pb,
                  "pp": (self.jac_pp_transport + j_rxn).tocsc()}
        return np.concatenate([g_b, g_p]), blocks

    def linear_solve(self, blocks, g):
        g_b, g_p = g[:self.n_b], g[self.n_b:]
        if not self.schur:  # full monolithic sparse solve of the same system
            full = bmat([[blocks["bb"], blocks["bp"]], [blocks["pb"], blocks["pp"]]]).tocsc()
            return spsolve(full, g)
        lu_pp = splu(blocks["pp"])           # block diagonal: one particle per bulk cell
        # Each particle couples only to its own bulk cell, so the columns of
        # J_pp^-1 J_pb have disjoint supports and J_bp J_pp^-1 J_pb is DIAGONAL.
        # One solve against the sum of the columns gives all of them at once.
        w = lu_pp.solve(blocks["pb"] @ np.ones(self.n_b))
        schur = (blocks["bb"] - diags_array(blocks["bp"] @ w)).tocsc()
        dx_b = spsolve(schur, g_b - blocks["bp"] @ lu_pp.solve(g_p))
        dx_p = lu_pp.solve(g_p - blocks["pb"] @ dx_b)
        return np.concatenate([dx_b, dx_p])

    def solve(self):
        x0 = np.ones(self.n_b + self.n_z * self.n_r)
        result = newton(self.residual, x0, solver=self.linear_solve, maxfev=50)
        if not result.success:
            raise RuntimeError(result.message)
        self.x = result.x
        return self

    def outlet(self):
        # the outlet face value, as the operator transports it (pitfalls P8)
        c_b = self.x[:self.n_b].reshape(self.shape_b)
        _, _, value_out, _ = compute_boundary_values(c_b, self.z_f, bc=self.bc_b)
        return float(np.ravel(value_out)[0])


def exact_outlet(alpha=0.5, phi=2.0, biot=5.0):
    g = phi / np.tanh(phi) - 1.0
    kappa = 3.0 * biot * g / (g + biot)
    return float(np.exp(-alpha * kappa))


def run_checks(verbose=True):
    results = {}
    exact = exact_outlet()
    values = [ParticleBed(n_z=n, n_r=60).solve().outlet() for n in (50, 100, 200, 400)]
    results["linear_outlet_rel_error_nz400"] = abs(values[-1] - exact) / exact
    errors = np.abs(np.array(values) - exact)
    results["grid_order_z"] = np.log2(errors[-2] / errors[-1])

    schur = ParticleBed(n_z=60, n_r=30, order=2).solve()
    full = ParticleBed(n_z=60, n_r=30, order=2, schur=False).solve()
    results["schur_vs_monolithic_nonlinear"] = float(np.max(np.abs(schur.x - full.x)))

    no_film = ParticleBed(n_z=100, n_r=60, biot=1e6).solve().outlet()
    results["break_row_no_film_shift"] = abs(no_film - values[1]) / values[1]

    passed = {
        "linear_outlet_rel_error_nz400": results["linear_outlet_rel_error_nz400"] < 5e-3,
        "grid_order_z": 0.7 < results["grid_order_z"] < 1.3,
        "schur_vs_monolithic_nonlinear": results["schur_vs_monolithic_nonlinear"] < 1e-10,
        "break_row_no_film_shift": results["break_row_no_film_shift"] > 0.05,
    }
    if verbose:
        print(f"exact outlet (linear) {exact:.6f}; n_z = 50..400: {np.round(values, 6)}")
        for key, value in results.items():
            print(f"{key:34s} = {value:.3e}   {'PASS' if passed[key] else 'FAIL'}")
    return results, passed


if __name__ == "__main__":
    run_checks()
