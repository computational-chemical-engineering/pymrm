"""Slurry bubble column: gas profiles and well-mixed liquid unknowns in one Newton system (S7).

Distilled from a Fischer-Tropsch slurry bubble column (Maretto and Krishna, 1999):
gas in plug flow along the column, a well-mixed slurry, gas-liquid transfer, a
contracting gas. Dimensionless, 0 < z < 1, species 0 = A (CO-like), 1 = B (H2-like).
All parameters are ILLUSTRATIVE, not the paper's.

    gas       d q_i/dz = -K_i (y_i - x_i)                       q = s(r) y,  r = w . y
    liquid    int_0^1 K_i (y_i - x_i) dz = Da_i x_A x_B          (transfer in = reaction)
    contraction  s = (1 + alpha) / (1 + alpha r)   (U / U_in, alpha < 0: moles lost)
    y_i(0) = 1 (feed), x_i = liquid concentration as equilibrium gas, relative to the feed

y is the gas concentration relative to its feed value, q the molar flow relative to the
feed. The velocity depends on the state, so the gas is convected as a MOLAR FLUX: the
upwind operator is built once with unit velocity and acts on q = s(r) y (the face flux
is the upstream cell's velocity times concentration). The Jacobian follows from the
product rule, dq/dy = s I + (ds/dr) y w^T per cell. Unknowns: y of shape (n, 2) plus two
liquid scalars, domains "gas" and "liq" of a BlockLayout(axis=None). Exact Jacobian,
no NumJac.

Outlet: {"outflow": True} and the outlet value read with compute_boundary_values, never
the last cell. An outlet bc of None is a closed end (a Dirichlet zero; pymrm 3.4.0
warns): on the gallery page (F2.3) it gave 95.63 % conversion where the outflow bc
gives the correct 93.31 %, with no error and a converged solve: the wrong state looked
better than the right one. (With the illustrative numbers here the
closed end has no steady state and newton fails, so no break row uses it.)

Checks, each able to fail:
1. Jacobian against finite differences with pymrm.checks.check_jacobian, away from the root.
2. Mole balance per species: feed = outlet flow + consumed by reaction, to round-off.
3. Closed form (alpha = 0): y_i = x_i + (1 - x_i) exp(-K_i z) turns the liquid balance
   into (1 - x_i)(1 - exp(-K_i)) = Da_i x_A x_B, solved with scipy.optimize.root
   (first-order upwind, n = 800: agreement within 0.5 %).
4. Break row: freezing the velocity at its inlet value (s = 1, the no-contraction
   shortcut) must move the conversion of A by more than 1 %.
"""

import numpy as np
from scipy.optimize import root
from scipy.sparse import csr_array, diags_array, eye_array, kron

from pymrm import (BlockLayout, compute_boundary_values, construct_convflux_upwind,
                   construct_div, newton)

try:
    from pymrm.checks import check_jacobian
except ImportError:  # pymrm 2.3.1 or older: the copy shipped with the conventions skill
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "conventions" / "scripts"))
    from pymrm_checks import check_jacobian

K = np.array([4.0, 6.0])      # transfer numbers k_L a H / (m U_in), illustrative
DA = np.array([3.0, 2.0])     # reaction numbers per species (nu_i / feed ratio), illustrative
W = np.array([1.0, 2.0]) / 3.0   # feed mole fractions, r = w . y; H2/CO = 2
ALPHA = -0.5                  # contraction factor at full conversion, illustrative


class SlurryColumn:
    def __init__(self, n=200, alpha=ALPHA, outflow=True, frozen=False):
        self.n, self.alpha, self.frozen = n, alpha, frozen
        self.z_f = np.linspace(0.0, 1.0, n + 1)
        self.z_c = 0.5 * (self.z_f[:-1] + self.z_f[1:])
        dz = np.diff(self.z_f)
        shape = (n, 2)
        self.layout = BlockLayout({"gas": shape, "liq": (2,)}, axis=None)
        # inlet z = 0: q = 1 (Dirichlet: a = 0, b = 1, d = 1; the bc acts on q, the
        # convected variable); outlet z = 1: the gas leaves with the last cell's flow
        # (None would be a closed end)
        self.bc = ({"a": 0.0, "b": 1.0, "d": 1.0}, {"outflow": True} if outflow else None)
        conv, conv_bc = construct_convflux_upwind(shape, self.z_f, self.z_c, self.bc, v=1.0)
        div = construct_div(shape, self.z_f, nu=0)
        self.conv_gas = (div @ conv).tocsr()
        self.g_bc = np.asarray((div @ conv_bc).todense()).ravel()
        # constant blocks, placed once
        place = self.layout.place
        k_cell = diags_array(np.tile(K, n))
        self.w_cell = kron(eye_array(n), csr_array(np.ones((2, 1)) * W), format="csr")  # row (k,i), col (k,j) = w_j
        self.lin = (place(k_cell, "gas", "gas")
                    + place(kron(csr_array(np.ones((n, 1))), csr_array(-np.diag(K))), "gas", "liq")
                    + place(kron(csr_array(dz[None, :]), csr_array(np.diag(K))), "liq", "gas")
                    + place(-np.diag(K), "liq", "liq"))
        self.dz = dz

    def flux(self, y):
        """q = s(r) y per cell and its Jacobian dq/dy (2 x 2 per cell), product rule."""
        if self.frozen:                        # break row: velocity frozen at its inlet value
            return y, eye_array(y.size, format="csr")
        r = y @ W
        den = 1.0 + self.alpha * r
        s, ds = (1.0 + self.alpha) / den, -self.alpha * (1.0 + self.alpha) / den**2
        return s[:, None] * y, diags_array(np.repeat(s, 2)) + diags_array((ds[:, None] * y).ravel()) @ self.w_cell

    def residual(self, u):
        lay, place = self.layout, self.layout.place
        y, x = lay.view(u, "gas"), lay.view(u, "liq")
        q, dq = self.flux(y)
        g = np.zeros(lay.size)
        lay.put(g, "gas", self.conv_gas @ q.ravel() + self.g_bc)
        lay.add(g, "gas", K * (y - x))
        sink = DA * x[0] * x[1]
        lay.put(g, "liq", K * (self.dz @ (y - x)) - sink)
        dsink = np.array([[DA[0] * x[1], DA[0] * x[0]], [DA[1] * x[1], DA[1] * x[0]]])
        jac = self.lin + place(self.conv_gas @ dq, "gas", "gas") + place(-dsink, "liq", "liq")
        return g, jac.tocsc()

    def solve(self):
        u0 = np.concatenate([np.ones(2 * self.n), [0.1, 0.1]])
        res = newton(self.residual, u0, tol=1e-12, maxfev=50)
        if not res.success:
            raise RuntimeError(res.message)
        self.u = res.x
        self.y = self.layout.view(self.u, "gas")
        self.x = self.layout.view(self.u, "liq")
        return self

    def outlet_flow(self):
        """Molar flow per species at z = 1 from compute_boundary_values, not the last cell."""
        q, _ = self.flux(self.y)
        value, _ = compute_boundary_values(q, self.z_f, self.z_c, bc=self.bc[1], axis=0, bound_id=1)
        return np.ravel(value)

    def conversion(self):
        """Fraction of the feed consumed by reaction, from the liquid balance."""
        return DA * self.x[0] * self.x[1]


def exact_alpha0():
    """Liquid concentrations for alpha = 0 from the closed-form gas profile."""
    f = 1.0 - np.exp(-K)

    def eq(x):
        return (1.0 - x) * f - DA * x[0] * x[1]
    return root(eq, [0.1, 0.1], tol=1e-14).x


def run_checks(verbose=True):
    results = {}
    m = SlurryColumn(200).solve()
    state = m.u * (1.0 + 0.05 * np.sin(np.arange(m.u.size)))
    results["jacobian_rel_error"] = check_jacobian(m.residual, state)["max_rel_error"]
    results["mole_balance_residual"] = np.max(np.abs(1.0 - m.outlet_flow() - m.conversion()))
    m0 = SlurryColumn(800, alpha=0.0).solve()
    x_exact = exact_alpha0()
    results["closed_form_rel_error"] = np.max(np.abs(m0.x - x_exact) / x_exact)
    base = m.conversion()[0]
    frozen = SlurryColumn(200, frozen=True).solve().conversion()[0]
    results["break_row_frozen_velocity_shift"] = abs(frozen - base) / base

    passed = {
        "jacobian_rel_error": results["jacobian_rel_error"] < 1e-5,
        "mole_balance_residual": results["mole_balance_residual"] < 1e-10,
        "closed_form_rel_error": results["closed_form_rel_error"] < 5e-3,
        "break_row_frozen_velocity_shift": results["break_row_frozen_velocity_shift"] > 0.01,
    }
    if verbose:
        print(f"conversion of A {base:.6f}, liquid x = {np.round(m.x, 6)}, "
              f"outlet flow = {np.round(m.outlet_flow(), 6)}")
        for key, value in results.items():
            print(f"{key:34s} = {value:.3e}   {'PASS' if passed[key] else 'FAIL'}")
    return results, passed


if __name__ == "__main__":
    run_checks()
