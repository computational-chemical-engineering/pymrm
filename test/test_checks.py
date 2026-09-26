import numpy as np
from scipy.sparse import csc_array

from pymrm import NumJac, construct_div, construct_grad, newton
from pymrm.checks import check_jacobian, find_roots, observed_orders, residual_check


def _pellet(n=40, phi2=4.0):
    x_f = np.linspace(0.0, 1.0, n + 1)
    shape = (n, 1)
    bc = ({"a": 1.0, "b": 0.0, "d": 0.0}, {"a": 0.0, "b": 1.0, "d": 1.0})
    grad, grad_bc = construct_grad(shape, x_f, bc=bc)
    div = construct_div(shape, x_f, nu=2)
    jac_const = (div @ -grad).tocsc()
    g_const = (div @ -grad_bc).toarray().reshape(-1, 1)
    numjac = NumJac(shape)

    def fun(c):
        g_r, j_r = numjac(lambda u: phi2 * u**2, c.reshape(shape))
        return g_const + jac_const @ c.reshape(-1, 1) + g_r.reshape(-1, 1), jac_const + j_r

    fun.jac_const = jac_const
    return fun, shape


def test_check_jacobian_accepts_correct_and_flags_wrong():
    fun, shape = _pellet()
    x = np.linspace(0.5, 1.0, shape[0]).reshape(shape)
    assert check_jacobian(fun, x)["ok"]

    def wrong(c):  # a forgotten term: the reaction Jacobian is missing
        g, _ = fun(c)
        return g, fun.jac_const
    bad = check_jacobian(wrong, x)
    assert not bad["ok"] and bad["worst_entry"][0] == bad["worst_entry"][1]


def test_observed_orders_second_order_quadrature():
    def midpoint(n):
        x = (np.arange(n) + 0.5) / n
        return np.mean(np.exp(x))
    out = observed_orders(midpoint, (10, 20, 40, 80))
    assert all(abs(p - 2.0) < 0.05 for p in out["orders"])
    assert abs(out["extrapolated"] - (np.e - 1.0)) < 1e-7


def test_find_roots_reports_all_and_failures():
    out = find_roots(lambda x: (x - 1.0) * (x - 2.0) * (x - 3.0), 0.0, 4.0, n_scan=41)
    assert np.allclose(out["roots"], [1.0, 2.0, 3.0]) and "multiplicity" in out["message"]
    none = find_roots(lambda x: x**2 + 1.0, -1.0, 1.0)
    assert none["roots"] == [] and "no sign change" in none["message"]

    def fragile(x):
        if x > 3.5:
            raise RuntimeError("solver failed")
        return x - 2.0
    out = find_roots(fragile, 0.0, 4.0, n_scan=41)
    assert np.allclose(out["roots"], [2.0]) and out["failed"]


def test_residual_check_is_scale_free():
    def trace(x):
        return np.array([x[0] ** 2 - 1e-20]), csc_array(np.array([[2.0 * x[0]]]))
    wrong = newton(trace, np.array([1e-8]))                          # 50x off, success=True
    assert not residual_check(trace, wrong.x)["ok"]
    right = newton(trace, np.array([1e-8]), tol=0.0, rtol=1e-12)
    assert residual_check(trace, right.x)["ok"]

    fun, shape = _pellet()
    sol = newton(fun, np.ones(shape))

    def scaled(c):  # the same equations in other units
        return tuple(v * 1e6 for v in fun(c))

    assert residual_check(fun, sol.x)["ok"] and residual_check(scaled, sol.x)["ok"]
