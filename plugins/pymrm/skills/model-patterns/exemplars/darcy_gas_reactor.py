"""Isothermal gas flow with Darcy pressure drop and reaction: pressure-velocity coupling (S10).

Dimensionless, 0 < z < 1, ideal gas with c_t = p (RT = 1), Darcy mobility 1:

    face velocity      u = -dp/dz
    total continuity   d(u c_t)/dz = 0            (no change in moles)
    species A          d(u c_A)/dz = -k c_A        (plug flow, first order)
    p(0) = p_in, p(1) = p_out;  c_A(0) = y_in p_in

Unknowns p and c_A in one Newton system, state (n, 2) = [p, c_A]. The velocity
depends on the state, so the convective flux is a product of two unknown
factors: F = u(p) * c_face(x). Both factors are linear in the state with
constant operators, so the Jacobian follows from the product rule with matrices
assembled once:

    dF/dx = diag(u) @ C          (C: upwind face interpolation, v = 1)
    dF/dp = diag(c_face) @ (-G)  (G: face gradient)

No NumJac is needed; this is the operator-sum style with a state-dependent
velocity (assembly-styles.md, "velocity varying along the reactor").

Checks, each able to fail:
1. Pressure profile against the exact p(z) = sqrt(p_in^2 - (p_in^2 - p_out^2) z)
   (second order).
2. Outlet conversion against the exact plug-flow result
   y_out / y_in = exp(-k / F_t * integral of p dz), with F_t = (p_in^2 - p_out^2) / 2.
3. Grid refinement of the conversion: observed order about 1 (upwind).
4. Jacobian against finite differences with pymrm.checks.check_jacobian
   (or the conventions skill's copy on an older pymrm).
5. Break row: freezing the velocity at its inlet value (incompressible
   assumption) must move the conversion by more than 5 %.
"""

import numpy as np
from scipy.sparse import diags_array

from pymrm import construct_convflux_upwind, construct_div, construct_grad, newton, update_array_indices

try:
    from pymrm.checks import check_jacobian
except ImportError:  # pymrm 2.3.1 or older: the copy shipped with the conventions skill
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "conventions" / "scripts"))
    from pymrm_checks import check_jacobian

P_IN, P_OUT, K_RXN, Y_IN = 1.0, 0.5, 1.0, 1.0


class DarcyGasReactor:
    def __init__(self, n_z=200, frozen_velocity=False):
        self.n = n_z
        self.frozen = frozen_velocity
        self.z_f = np.linspace(0.0, 1.0, n_z + 1)
        self.z_c = 0.5 * (self.z_f[:-1] + self.z_f[1:])
        self.shape = (n_z, 2)          # fields: 0 = p, 1 = c_A
        field = (n_z, 1)
        # pressure: fixed at both ends (a = 0, b = 1)
        bc_p = ({"a": 0.0, "b": 1.0, "d": P_IN}, {"a": 0.0, "b": 1.0, "d": P_OUT})
        grad, grad_bc = construct_grad(field, self.z_f, self.z_c, bc_p)
        self.grad, self.grad_bc = grad.tocsr(), grad_bc.toarray().ravel()
        self.div = construct_div(field, self.z_f, nu=0)
        # face interpolation (upwind for u > 0) of p and of c_A, each with its inlet value
        conv_p, conv_p_bc = construct_convflux_upwind(field, self.z_f, self.z_c, bc_p, v=1.0)
        bc_c = ({"a": 0.0, "b": 1.0, "d": Y_IN * P_IN}, {"a": 1.0, "b": 0.0, "d": 0.0})
        conv_c, conv_c_bc = construct_convflux_upwind(field, self.z_f, self.z_c, bc_c, v=1.0)
        self.interp = [(conv_p.tocsr(), conv_p_bc.toarray().ravel()),
                       (conv_c.tocsr(), conv_c_bc.toarray().ravel())]
        self.reaction = K_RXN * diags_array(np.ones(n_z))
        self.field = field
        self.u_inlet = (P_IN**2 - P_OUT**2) / 2.0 / P_IN   # exact inlet velocity

    def place(self, block, i, j):
        """Put a field-by-field block into rows of field i and columns of field j."""
        return update_array_indices(block, (self.field, self.field), self.shape, offset=((0, i), (0, j)))

    def velocity(self, p):
        return -(self.grad @ p + self.grad_bc)

    def species_velocity(self, p):
        """Break row: the incompressible shortcut freezes the species velocity."""
        return np.full(self.n + 1, self.u_inlet) if self.frozen else self.velocity(p)

    def residual(self, state):
        state = state.reshape(self.shape)
        p = state[:, 0]
        u = self.velocity(p)
        g = np.empty(self.shape)
        blocks = []
        for i, x in enumerate((p, state[:, 1])):
            conv, conv_bc = self.interp[i]
            face = conv @ x + conv_bc
            u_i = u if i == 0 else self.species_velocity(p)
            g[:, i] = self.div @ (u_i * face)
            blocks.append((i, i, self.div @ diags_array(u_i) @ conv))
            if i == 0 or not self.frozen:
                blocks.append((i, 0, self.div @ diags_array(face) @ (-self.grad)))
        g[:, 1] += K_RXN * state[:, 1]
        blocks.append((1, 1, self.reaction))
        jac = sum(self.place(block, i, j) for i, j, block in blocks)
        return g.reshape(-1, 1), jac.tocsc()

    def solve(self):
        guess = np.empty(self.shape)
        guess[:, 0] = P_IN + (P_OUT - P_IN) * self.z_c
        guess[:, 1] = Y_IN * guess[:, 0]
        result = newton(self.residual, guess, maxfev=50)
        if not result.success:
            raise RuntimeError(result.message)
        self.state = result.x.reshape(self.shape)
        return self

    def conversion(self):
        p = self.state[:, 0]
        u = self.species_velocity(p)
        conv_c, conv_c_bc = self.interp[1]
        flux_out = u[-1] * (conv_c @ self.state[:, 1] + conv_c_bc)[-1]
        return 1.0 - flux_out / (u[0] * Y_IN * P_IN)


def exact_pressure(z):
    return np.sqrt(P_IN**2 - (P_IN**2 - P_OUT**2) * z)


def exact_conversion():
    a, b = P_IN**2, P_IN**2 - P_OUT**2
    integral_p = 2.0 / (3.0 * b) * (a**1.5 - (a - b) ** 1.5)
    total_flux = b / 2.0
    return 1.0 - np.exp(-K_RXN / total_flux * integral_p)


def run_checks(verbose=True):
    results = {}
    m = DarcyGasReactor(400).solve()
    results["pressure_max_error_n400"] = np.max(np.abs(m.state[:, 0] - exact_pressure(m.z_c)))
    x_exact = exact_conversion()
    values = [DarcyGasReactor(n).solve().conversion() for n in (100, 200, 400, 800)]
    results["conversion_rel_error_n800"] = abs(values[-1] - x_exact) / x_exact
    errors = np.abs(np.array(values) - x_exact)
    results["grid_order"] = np.log2(errors[-2] / errors[-1])
    state = m.state * (1.0 + 0.05 * np.sin(np.arange(m.state.size)).reshape(m.shape))
    results["jacobian_rel_error"] = check_jacobian(m.residual, state)["max_rel_error"]
    frozen = DarcyGasReactor(400, frozen_velocity=True).solve().conversion()
    results["break_row_frozen_velocity_shift"] = abs(frozen - x_exact) / x_exact

    passed = {
        "pressure_max_error_n400": results["pressure_max_error_n400"] < 1e-4,
        "conversion_rel_error_n800": results["conversion_rel_error_n800"] < 2e-3,
        "grid_order": 0.8 < results["grid_order"] < 1.2,
        "jacobian_rel_error": results["jacobian_rel_error"] < 1e-5,
        "break_row_frozen_velocity_shift": results["break_row_frozen_velocity_shift"] > 0.05,
    }
    if verbose:
        print(f"exact conversion {x_exact:.6f}; n = 100..800: {np.round(values, 6)}")
        for key, value in results.items():
            print(f"{key:34s} = {value:.3e}   {'PASS' if passed[key] else 'FAIL'}")
    return results, passed


if __name__ == "__main__":
    run_checks()
