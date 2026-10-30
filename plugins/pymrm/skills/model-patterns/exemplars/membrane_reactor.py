"""Countercurrent membrane reactor, monolithic block assembly (structures S3, S7).

Two channels along 0 < z < 1 (dimensionless), solved as ONE Newton system:

    retentate, flows +z:  dc/dt + d(c)/dz = (1/Pe) d2c/dz2 + R(c) - J
        species A, B; reaction A -> B, rate Da * c_A (first order)
    permeate, flows -z:   dp/dt - d(p)/dz = (1/Pe) d2p/dz2 + J / sigma
        species B only
    membrane flux of B (per retentate volume): J = St * (c_B - p_B)
    sigma = permeate/retentate volume ratio

Boundary conditions (Danckwerts at each inlet, zero gradient at each outlet):
    retentate z = 0: c - (1/Pe) dc/dz = c_in ;  z = 1: dc/dz = 0
    permeate  z = 1: p + (1/Pe) dp/dz = 0 (sweep enters clean, flows -z) ; z = 0: dp/dz = 0

Assembly (style B in assembly-styles.md): each channel has its own operators on
its own sub-shape ((n, 2) retentate, (n, 1) permeate); the monolithic state is
(n, 3) with fields [A_ret, B_ret, B_perm], the two channels concatenated along
the field axis (BlockLayout(..., axis=-1)). Constant blocks (transport, reaction,
membrane) are placed ONCE with layout.assemble. The model is linear, so
the steady Jacobian is constant and one sparse solve gives the answer.

Checks, each able to fail:
1. Plug-flow limit (Pe = 1e4) against an independent route: the countercurrent
   two-point boundary-value ODE solved with scipy solve_bvp (no pymrm). At this
   Pe the upwind scheme's numerical diffusion exceeds the physical one, so this
   checks that the discretisation converges to plug flow (error O(dz)).
All checks run at a permeate/retentate volume ratio sigma = 0.5: at sigma = 1 a
J * sigma versus J / sigma error is invisible to every check.
2. Species balance: A fed = A out + A reacted; B produced = B out of both
   channels. Evaluated with face values; with the operators used here it is
   structural (exact by construction) and reported as a code check.
3. Grid refinement of the retentate B outlet: observed order about 1 (upwind).
4. Break row: reversing the permeate flow direction (co-current) must move the
   permeate B outlet, proving the coupling direction is seen by the checks.
"""

import numpy as np
from scipy.integrate import solve_bvp
from scipy.sparse.linalg import spsolve

from pymrm import (compute_boundary_values, construct_coefficient_matrix, construct_convflux_upwind,
                   construct_div, construct_grad, BlockLayout)


class MembraneReactor:
    """Steady countercurrent membrane reactor, both channels in one linear system."""

    def __init__(self, n_z=200, peclet=50.0, damkohler=2.0, stanton=3.0, sigma=1.0,
                 c_in=(1.0, 0.0), permeate_velocity=-1.0):
        self.n_z = n_z
        self.peclet, self.damkohler, self.stanton, self.sigma = peclet, damkohler, stanton, sigma
        self.c_in = np.asarray(c_in, dtype=float)
        self.v_perm = permeate_velocity
        self.z_f = np.linspace(0.0, 1.0, n_z + 1)
        self.z_c = 0.5 * (self.z_f[:-1] + self.z_f[1:])
        self.shape = (n_z, 3)            # fields: A_ret, B_ret, B_perm
        self.shape_ret = (n_z, 2)
        self.shape_perm = (n_z, 1)
        # the state is np.concatenate([retentate, permeate], axis=-1)
        self.layout = BlockLayout({"retentate": self.shape_ret, "permeate": self.shape_perm}, axis=-1)
        self._build()

    def _channel(self, shape, bc, velocity):
        """Transport operator of one channel: div(v c - D grad c) as matrix + bc vector."""
        grad, grad_bc = construct_grad(shape, self.z_f, self.z_c, bc)
        conv, conv_bc = construct_convflux_upwind(shape, self.z_f, self.z_c, bc, v=velocity)
        div = construct_div(shape, self.z_f, nu=0)          # nu=0: plane channel
        mat = div @ (conv - grad / self.peclet)
        vec = (div @ (conv_bc - grad_bc / self.peclet)).toarray().ravel()
        return mat, vec, div, conv, conv_bc, grad, grad_bc

    def _build(self):
        pe = self.peclet
        # retentate: z = 0: c - (1/Pe) dc/dz = c_in (outward normal -z, so a = +1/Pe)
        #            z = 1: dc/dz = 0
        bc_ret = ({"a": 1.0 / pe, "b": 1.0, "d": self.c_in.reshape(1, 2)},
                  {"a": 1.0, "b": 0.0, "d": 0.0})
        # permeate flowing -z: inlet at z = 1: p + (1/Pe) dp/dz = 0 (outward +z, a = +1/Pe)
        #                     outlet at z = 0: dp/dz = 0
        bc_perm = ({"a": 1.0, "b": 0.0, "d": 0.0}, {"a": 1.0 / pe, "b": 1.0, "d": 0.0})
        if self.v_perm > 0:  # break row: co-current permeate, inlet at z = 0
            bc_perm = ({"a": 1.0 / pe, "b": 1.0, "d": 0.0}, {"a": 1.0, "b": 0.0, "d": 0.0})
        self.bc_ret, self.bc_perm = bc_ret, bc_perm

        t_ret, g_ret, *_ = self._channel(self.shape_ret, bc_ret, 1.0)
        t_perm, g_perm, *_ = self._channel(self.shape_perm, bc_perm, self.v_perm)

        n = self.n_z
        # local (per-cell) coupling blocks: one small matrix per cell, rows the fields of
        # the row channel, columns those of the column channel, spread over all cells by
        # construct_coefficient_matrix with shapes ((n, rows, 1), (n, 1, cols)). Rows of the
        # residual: transport + consumption - production + J (retentate), transport - J/sigma
        # (permeate), with J = St (c_B - p_B). No flat index is computed by hand.
        da, st, sg = self.damkohler, self.stanton, self.sigma

        def per_cell(matrix):
            matrix = np.asarray(matrix, dtype=float)
            n_rows, n_cols = matrix.shape
            return construct_coefficient_matrix(matrix, shape=((n, n_rows, 1), (n, 1, n_cols)))

        local_ret = per_cell([[da, 0.0],       # A_ret row: A consumed
                              [-da, st]])      # B_ret row: B produced from A, +St c_B
        ret_perm = per_cell([[0.0], [-st]])    # B_ret row: -St p_B
        perm_ret = per_cell([[0.0, -st / sg]])  # B_perm row: -(St/sigma) c_B
        local_perm = per_cell([[st / sg]])     # B_perm row: +(St/sigma) p_B
        self.jac = self.layout.assemble([
            (t_ret + local_ret, "retentate", "retentate"),
            (t_perm + local_perm, "permeate", "permeate"),
            (ret_perm, "retentate", "permeate"),
            (perm_ret, "permeate", "retentate"),
        ])
        self.g_const = np.zeros(self.shape)
        self.layout.put(self.g_const, "retentate", g_ret)
        self.layout.put(self.g_const, "permeate", g_perm)

    def solve(self):
        self.u = spsolve(self.jac, -self.g_const.ravel()).reshape(self.shape)
        return self

    def outlets(self):
        """Face values at each channel's outlet: retentate at z = 1, permeate at its exit."""
        ret, perm = self.layout.view(self.u, "retentate"), self.layout.view(self.u, "permeate")
        _, _, ret_out, _ = compute_boundary_values(ret, self.z_f, self.z_c, self.bc_ret)
        perm_lo, _, perm_hi, _ = compute_boundary_values(perm, self.z_f, self.z_c, self.bc_perm)
        perm_out = perm_lo if self.v_perm < 0 else perm_hi
        return np.ravel(ret_out), float(np.ravel(perm_out)[0])


def plug_flow_bvp(damkohler=2.0, stanton=3.0, sigma=1.0, c_in=(1.0, 0.0)):
    """Independent route: plug-flow countercurrent ODEs as a two-point BVP (no pymrm)."""
    def rhs(z, y):
        a, b, p = y
        flux = stanton * (b - p)
        return np.vstack([-damkohler * a, damkohler * a - flux, -flux / sigma])

    def bc(ya, yb):
        return np.array([ya[0] - c_in[0], ya[1] - c_in[1], yb[2]])

    z = np.linspace(0.0, 1.0, 101)
    sol = solve_bvp(rhs, bc, z, np.zeros((3, z.size)), tol=1e-10, max_nodes=100000)
    assert sol.success, sol.message
    return sol.y[:, -1], sol.y[2, 0]   # retentate outlet (z = 1), permeate outlet (z = 0)


def run_checks(verbose=True):
    results = {}
    ref_ret, ref_perm = plug_flow_bvp(sigma=0.5)
    model = MembraneReactor(n_z=1600, peclet=1e4, sigma=0.5).solve()
    ret_out, perm_out = model.outlets()
    results["plug_flow_vs_bvp_retentate_B"] = abs(ret_out[1] - ref_ret[1])
    results["plug_flow_vs_bvp_permeate_B"] = abs(perm_out - ref_perm)

    base = MembraneReactor(n_z=200, sigma=0.5).solve()
    ret_out, perm_out = base.outlets()
    dz = np.diff(base.z_f)
    reacted = base.damkohler * np.sum(base.u[:, 0] * dz)
    results["structural_balance_A"] = abs(base.c_in[0] - ret_out[0] - reacted)
    results["structural_balance_B"] = abs(reacted - ret_out[1] - base.sigma * perm_out)

    values = [MembraneReactor(n_z=n, sigma=0.5).solve().outlets()[0][1] for n in (100, 200, 400, 800)]
    diffs = np.abs(np.diff(values))
    results["grid_order"] = np.log2(diffs[-2] / diffs[-1])

    cocurrent = MembraneReactor(n_z=200, sigma=0.5, permeate_velocity=1.0).solve().outlets()[1]
    results["break_row_cocurrent_shift"] = abs(cocurrent - perm_out) / perm_out

    passed = {
        "plug_flow_vs_bvp_retentate_B": results["plug_flow_vs_bvp_retentate_B"] < 2e-3,
        "plug_flow_vs_bvp_permeate_B": results["plug_flow_vs_bvp_permeate_B"] < 2e-3,
        "structural_balance_A": results["structural_balance_A"] < 1e-10,
        "structural_balance_B": results["structural_balance_B"] < 1e-10,
        "grid_order": 0.8 < results["grid_order"] < 1.2,
        "break_row_cocurrent_shift": results["break_row_cocurrent_shift"] > 0.05,
    }
    if verbose:
        print(f"plug-flow reference: retentate B out {ref_ret[1]:.6f}, permeate B out {ref_perm:.6f}")
        print(f"base case (Pe=50, n=200): retentate out {ret_out}, permeate B out {perm_out:.6f}")
        for key, value in results.items():
            print(f"{key:34s} = {value:.3e}   {'PASS' if passed[key] else 'FAIL'}")
    return results, passed


if __name__ == "__main__":
    run_checks()
