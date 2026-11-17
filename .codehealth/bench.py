"""Benchmark workloads for the codehealth gate: ``python .codehealth/bench.py <case>``.

Each case exercises the code that runs per residual or per Newton iteration, where an
extracted helper adds call overhead, or the assembly code that a refactor touches most.
The gate times the whole command (ABBA pairs, base against head), so each case runs
long enough (about 1 to 2 s) that interpreter start-up is a small share.

Cases:
  newton        small non-isothermal pellet solved repeatedly (newton loop overhead, NumJac),
                plus a natural-parameter continuation sweep
  tvd           TVD convection-diffusion-reaction steady state (ConvFluxTVD.__call__,
                interp_cntr_to_stagg_tvd, newton), plus compute_boundary_values calls
  ibm           immersed-boundary assembly for three particles (construct_ibm_particles,
                the IBM side builders, reconstruction)
  assembly      operator assembly with boundary and interface conditions (construct_grad_bc,
                construct_convflux_bc, construct_interface_matrices, construct_coefficient_matrix)
"""
import sys
import warnings

import numpy as np

import pymrm
from pymrm import (NumJac, construct_grad, construct_div, newton, continuation, ConvFluxTVD,
                   compute_boundary_values, construct_convflux_upwind, construct_interface_matrices,
                   construct_coefficient_matrix)


def _pellet(phi, n=30, gamma=20.0, beta=0.6):
    shape = (n, 1)
    r_f = np.linspace(0.0, 1.0, n + 1)
    bc = ({"a": 1, "b": 0, "d": 0}, {"a": 0, "b": 1, "d": 1})
    grad, grad_bc = construct_grad(shape, r_f, bc=bc)
    div = construct_div(shape, r_f, nu=2)
    jac_const = -(div @ grad)
    g_const = -(div @ grad_bc)
    g_const = np.asarray(g_const.toarray() if hasattr(g_const, "toarray") else g_const).reshape(-1, 1)
    numjac = NumJac(shape)

    def rate(c, phi=phi):
        t = beta * (1.0 - c)
        return phi**2 * c * np.exp(gamma * t / (1.0 + t))

    def fun(c, phi=phi):
        g_r, j_r = numjac(lambda cc: rate(cc, phi), c.reshape(shape))
        return g_const + jac_const @ c.reshape(-1, 1) + g_r.reshape(-1, 1), jac_const + j_r

    return fun, shape


def case_newton():
    fun, shape = _pellet(0.5)
    for _ in range(60):
        sol = newton(fun, np.ones(shape))
        assert sol.success
    fun2, shape2 = _pellet(0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(2):
            continuation(fun2, np.ones(shape2), np.linspace(0.1, 0.6, 11))


def case_tvd():
    n, u, D, k = 200, 1.0, 1e-3, 2.0
    x_f = np.linspace(0, 1, n + 1)
    bc_conv = ({"a": 0, "b": 1, "d": 1.0}, {"outflow": True})
    bc_diff = ({"a": 0, "b": 1, "d": 1.0}, {"a": 1, "b": 0, "d": 0})
    tvd = ConvFluxTVD((n,), x_f, bc=bc_conv, tvd_limiter=pymrm.vanleer)
    grad, grad_bc = construct_grad((n,), x_f, bc=bc_diff)
    div = construct_div((n,), x_f)
    diff, diff_bc = -D * (div @ grad), -D * (div @ grad_bc).toarray().ravel()
    eye = np.eye(n)

    def model(c):
        flux, jac = tvd(c, u)
        return div @ flux + diff @ c + diff_bc + k * c, div @ jac + diff + k * eye

    for _ in range(30):
        sol = newton(model, np.zeros(n), tol=1e-12, maxfev=40, damping="monotone")
        assert sol.success
    c = sol.x.reshape(-1, 1)
    for _ in range(2000):
        tvd(sol.x, u)
        compute_boundary_values(c, x_f, bc=bc_diff)


def case_ibm():
    from pymrm.particles import Sphere
    from pymrm import construct_ibm_particles, construct_ibm, construct_ibm_normal_derivative
    N = 128
    x_f = np.linspace(0.0, 1.0, N + 1)
    x_c = 0.5 * (x_f[:-1] + x_f[1:])
    parts = [Sphere((0.303, 0.297), 0.18), Sphere((0.703, 0.351), 0.15), Sphere((0.48, 0.72), 0.2)]
    for _ in range(100):
        construct_ibm_particles(parts, (x_c, x_c))
    X, Y = np.meshgrid(x_c, x_c, indexing="ij")
    sdf = np.sqrt((X - 0.503) ** 2 + (Y - 0.497) ** 2) - 0.28
    ibm = construct_ibm(sdf, [x_c, x_c])
    for _ in range(3):
        construct_ibm_normal_derivative(ibm, sdf, [x_c, x_c])


def case_assembly():
    shape = (40, 30, 2)
    x_f = np.linspace(0, 1, 41)
    bc = ({"a": 1, "b": 0.5, "d": 1.0}, {"a": 0, "b": 1, "d": [0.0, 1.0]})
    for _ in range(60):
        construct_grad(shape, x_f, bc=bc)
        construct_convflux_upwind(shape, x_f, bc=bc, v=1.0)
        construct_interface_matrices(((40, 2), (30, 2)), (np.linspace(0, 1, 41), np.linspace(1, 2, 31)))
        construct_coefficient_matrix(np.ones((40, 30, 2)))


CASES = {"newton": case_newton, "tvd": case_tvd, "ibm": case_ibm, "assembly": case_assembly}

if __name__ == "__main__":
    CASES[sys.argv[1]]()
