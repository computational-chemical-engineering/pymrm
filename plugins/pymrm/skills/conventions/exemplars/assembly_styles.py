"""One model, three ways to assemble its Jacobian (see references/assembly-styles.md).

Model: steady nonisothermal first-order reaction in a spherical pellet,
dimensionless (radius 1, surface values 1):

    div(-grad c) + phi^2 c exp(gamma (1 - 1/T)) = 0
    div(-grad T) - beta phi^2 c exp(gamma (1 - 1/T)) = 0
    r = 0: dc/dr = dT/dr = 0;   r = 1: c = T = 1

Style A, operator sum: both fields in one (n, 2) state; the transport Jacobian
    comes from the operators, assembled once; NumJac differentiates only the
    pointwise source (2 x 2 blocks per cell). Jacobian = constant + local.
Style B, block assembly: each field and each coupling is its own block on its
    own sub-shape (here (n, 1)), with analytic local derivatives, placed into the
    monolithic state with update_array_indices. This is how multi-field and
    multi-domain models with different operators per field are built.
Style B, fixed pattern: the same blocks, but the placed sparsity pattern is
    computed once and each call only scatters values into it: the fastest
    variant when local derivatives are analytic.
Style C, full-residual NumJac: NumJac differentiates the whole residual; its
    stencil must cover every coupling (neighbours along r, both fields).

Checks, each able to fail:
1. The three styles give the same solution.
2. The three Jacobians agree at a common state (C only to finite-difference
   accuracy).
3. The Prater relation T - 1 = beta (1 - c) holds. With one operator for both
   fields it is a STRUCTURAL identity of this discretisation, so it is reported
   as a consistency check, not as evidence of accuracy.
4. Break row: style C with the default stencil (no neighbour coupling) has a
   wrong Jacobian; Newton then fails or needs many more iterations.
Timings per residual-and-Jacobian evaluation are printed, not asserted.
"""

import time
import warnings

import numpy as np
from scipy.sparse import csc_array, diags_array

from pymrm import NumJac, construct_div, construct_grad, newton, update_array_indices

PHI2, BETA, GAMMA = 0.25, 0.3, 20.0


def _grid(n_r):
    r_f = np.linspace(0.0, 1.0, n_r + 1)
    return r_f, 0.5 * (r_f[:-1] + r_f[1:])


def _rate(c, temp):
    return PHI2 * c * np.exp(GAMMA * (1.0 - 1.0 / temp))


class OperatorSum:
    """Style A: constant transport Jacobian plus NumJac on the local source."""

    def __init__(self, n_r=200):
        self.shape = (n_r, 2)  # fields: 0 = c, 1 = T
        r_f, r_c = _grid(n_r)
        # r = 0: symmetry for both fields; r = 1: c = 1 and T = 1
        bc = ({"a": [1.0, 1.0], "b": [0.0, 0.0], "d": [0.0, 0.0]},
              {"a": [0.0, 0.0], "b": [1.0, 1.0], "d": [1.0, 1.0]})
        grad, grad_bc = construct_grad(self.shape, r_f, r_c, bc)
        div = construct_div(self.shape, r_f, nu=2)  # nu=2: sphere
        self.jac_const = (div @ -grad).tocsc()
        self.g_const = (div @ -grad_bc).toarray().reshape((-1, 1))
        self.numjac = NumJac(self.shape)

    @staticmethod
    def source(u):
        r = _rate(u[..., 0], u[..., 1])
        return np.stack([r, -BETA * r], axis=-1)

    def residual(self, u):
        g_src, jac_src = self.numjac(self.source, u.reshape(self.shape))
        g = self.g_const + self.jac_const @ u.reshape((-1, 1)) + g_src.reshape((-1, 1))
        return g, self.jac_const + jac_src


class BlockAssembly:
    """Style B: one block per field and per coupling, placed with update_array_indices."""

    def __init__(self, n_r=200):
        self.n_r = n_r
        self.shape = (n_r, 2)
        self.field_shape = (n_r, 1)
        r_f, r_c = _grid(n_r)
        div = construct_div(self.field_shape, r_f, nu=2)  # nu=2: sphere
        # each field gets its own operator; here they happen to be equal
        # r = 0: dc/dr = 0; r = 1: c = 1   (same for T)
        bc = ({"a": 1.0, "b": 0.0, "d": 0.0}, {"a": 0.0, "b": 1.0, "d": 1.0})
        grad_c, grad_c_bc = construct_grad(self.field_shape, r_f, r_c, bc)
        grad_t, grad_t_bc = construct_grad(self.field_shape, r_f, r_c, bc)
        self.jac_cc_transport = (div @ -grad_c).tocsc()
        self.jac_tt_transport = (div @ -grad_t).tocsc()
        self.g_c_bc = (div @ -grad_c_bc).toarray().ravel()
        self.g_t_bc = (div @ -grad_t_bc).toarray().ravel()
        self.offset_c = (0, 0)
        self.offset_t = (0, 1)
        # constant blocks are placed into the monolithic state ONCE
        self.jac_const = (self.place(self.jac_cc_transport, self.offset_c, self.offset_c)
                          + self.place(self.jac_tt_transport, self.offset_t, self.offset_t)).tocsc()

    def place(self, block, rows, cols):
        """Put an (n, 1) x (n, 1) block into the rows of field `rows` and columns of field `cols`."""
        return update_array_indices(block, (self.field_shape, self.field_shape), self.shape,
                                    offset=(rows, cols))

    def residual(self, u):
        u = u.reshape(self.shape)
        c, temp = u[:, 0], u[:, 1]
        r = _rate(c, temp)
        dr_dc = PHI2 * np.exp(GAMMA * (1.0 - 1.0 / temp))
        dr_dt = r * GAMMA / temp**2
        g = np.empty(self.shape)
        g[:, 0] = self.jac_cc_transport @ c + self.g_c_bc + r
        g[:, 1] = self.jac_tt_transport @ temp + self.g_t_bc - BETA * r
        # state-dependent blocks: re-placed every call
        jac = (self.jac_const
               + self.place(diags_array(dr_dc), self.offset_c, self.offset_c)
               + self.place(diags_array(dr_dt), self.offset_c, self.offset_t)
               + self.place(diags_array(-BETA * dr_dc), self.offset_t, self.offset_c)
               + self.place(diags_array(-BETA * dr_dt), self.offset_t, self.offset_t))
        return g.reshape((-1, 1)), jac.tocsc()


class BlockPattern(BlockAssembly):
    """Style B, fast: the placed sparsity pattern is computed once; each call only
    scatters new values into it. No index remapping and no sparse sums per call."""

    def __init__(self, n_r=200):
        super().__init__(n_r)
        n_cells = np.arange(n_r)
        flat_c, flat_t = 2 * n_cells, 2 * n_cells + 1  # flat index of (cell, field) in (n, 2)
        const = self.jac_const.tocoo()
        # every entry that will ever be written: constant part, then the four local blocks
        rows = np.concatenate([const.row, flat_c, flat_c, flat_t, flat_t])
        cols = np.concatenate([const.col, flat_c, flat_t, flat_c, flat_t])
        n_rows = self.jac_const.shape[0]
        keys = cols.astype(np.int64) * n_rows + rows  # CSC order: by column, then row
        unique_keys, self.position = np.unique(keys, return_inverse=True)
        self.indices = (unique_keys % n_rows).astype(np.int32)
        self.indptr = np.searchsorted(unique_keys // n_rows, np.arange(n_rows + 1)).astype(np.int32)
        self.nnz = unique_keys.size
        self.const_values = const.data

    def residual(self, u):
        u = u.reshape(self.shape)
        c, temp = u[:, 0], u[:, 1]
        r = _rate(c, temp)
        dr_dc = PHI2 * np.exp(GAMMA * (1.0 - 1.0 / temp))
        dr_dt = r * GAMMA / temp**2
        g = np.empty(self.shape)
        g[:, 0] = self.jac_cc_transport @ c + self.g_c_bc + r
        g[:, 1] = self.jac_tt_transport @ temp + self.g_t_bc - BETA * r
        values = np.concatenate([self.const_values, dr_dc, dr_dt, -BETA * dr_dc, -BETA * dr_dt])
        data = np.bincount(self.position, weights=values, minlength=self.nnz)
        return g.reshape((-1, 1)), csc_array((data, self.indices, self.indptr), shape=self.jac_const.shape)


class FullResidual:
    """Style C: NumJac on the whole residual; the stencil must cover every coupling."""

    def __init__(self, n_r=200, complete_stencil=True):
        self.base = OperatorSum(n_r)
        self.shape = self.base.shape
        stencil = {"axes_diagonals": [0], "axes_blocks": [-1]} if complete_stencil else {}
        self.numjac = NumJac(self.shape, **stencil)

    def full(self, u):
        transport = (self.base.g_const + self.base.jac_const @ u.reshape((-1, 1))).reshape(self.shape)
        return transport + OperatorSum.source(u)

    def residual(self, u):
        g, jac = self.numjac(self.full, u.reshape(self.shape))
        return g.reshape((-1, 1)), jac


def solve(model, maxfev=50):
    result = newton(model.residual, np.ones(model.shape), maxfev=maxfev)
    return result.x.reshape(model.shape), result


def time_residual(model, u, repeats=5):
    start = time.perf_counter()
    for _ in range(repeats):
        model.residual(u)
    return (time.perf_counter() - start) / repeats


def run_checks(verbose=True, n_r=200):
    models = {"A operator sum": OperatorSum(n_r), "B block assembly": BlockAssembly(n_r),
              "B block, fixed pattern": BlockPattern(n_r), "C full-residual NumJac": FullResidual(n_r)}
    solutions = {name: solve(model) for name, model in models.items()}
    u_a = solutions["A operator sum"][0]

    results = {
        "max_solution_diff_B": np.max(np.abs(solutions["B block assembly"][0] - u_a)),
        "max_solution_diff_C": np.max(np.abs(solutions["C full-residual NumJac"][0] - u_a)),
        "max_solution_diff_B_pattern": np.max(np.abs(solutions["B block, fixed pattern"][0] - u_a)),
    }
    u_test = 0.5 * (u_a + 1.0)  # a common state away from the solution
    jacs = {name: model.residual(u_test)[1].toarray() for name, model in models.items()}
    scale = np.max(np.abs(jacs["B block assembly"]))
    results["jac_diff_A_vs_B"] = np.max(np.abs(jacs["A operator sum"] - jacs["B block assembly"])) / scale
    results["jac_diff_Bpattern_vs_B"] = np.max(np.abs(jacs["B block, fixed pattern"]
                                                      - jacs["B block assembly"])) / scale
    results["jac_diff_C_vs_B"] = np.max(np.abs(jacs["C full-residual NumJac"] - jacs["B block assembly"])) / scale
    results["prater_structural_residual"] = np.max(np.abs((u_a[:, 1] - 1.0) - BETA * (1.0 - u_a[:, 0])))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a singular Jacobian counts as failure
            _, broken = solve(FullResidual(n_r, complete_stencil=False))
        broken_iterations = broken.nit if broken.success else 10**6
    except (ValueError, RuntimeError, np.linalg.LinAlgError, Warning):
        broken_iterations = 10**6  # singular or non-finite Newton step
    results["break_row_incomplete_stencil_iterations"] = float(broken_iterations)
    results["reference_iterations"] = float(solutions["A operator sum"][1].nit)

    passed = {
        "max_solution_diff_B": results["max_solution_diff_B"] < 1e-10,
        "max_solution_diff_C": results["max_solution_diff_C"] < 1e-8,
        "jac_diff_A_vs_B": results["jac_diff_A_vs_B"] < 1e-5,
        "max_solution_diff_B_pattern": results["max_solution_diff_B_pattern"] < 1e-10,
        "jac_diff_Bpattern_vs_B": results["jac_diff_Bpattern_vs_B"] < 1e-12,
        "jac_diff_C_vs_B": results["jac_diff_C_vs_B"] < 1e-5,
        "prater_structural_residual": results["prater_structural_residual"] < 1e-10,
        "break_row_incomplete_stencil_iterations":
            results["break_row_incomplete_stencil_iterations"] > 3 * results["reference_iterations"],
        "reference_iterations": True,
    }
    if verbose:
        print(f"c at centre: {u_a[0, 0]:.10f}, T at centre: {u_a[0, 1]:.10f}")
        for name, model in models.items():
            print(f"{name:24s} residual+Jacobian: {1e3 * time_residual(model, u_a):7.3f} ms, "
                  f"Newton iterations: {solutions[name][1].nit}")
        for key, value in results.items():
            print(f"{key:40s} = {value:.3e}   {'PASS' if passed[key] else 'FAIL'}")
    return results, passed


if __name__ == "__main__":
    run_checks()
