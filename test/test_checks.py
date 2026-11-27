import numpy as np
import pytest
from scipy.sparse import csc_array, diags_array

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
    wrong = newton(trace, np.array([1e-8]), damping="none", backward_tol=None, stagnation=False)  # 50x off, success=True
    assert not residual_check(trace, wrong.x)["ok"]
    right = newton(trace, np.array([1e-8]), tol=0.0, rtol=1e-12)
    assert residual_check(trace, right.x)["ok"]

    fun, shape = _pellet()
    sol = newton(fun, np.ones(shape))

    def scaled(c):  # the same equations in other units
        return tuple(v * 1e6 for v in fun(c))

    assert residual_check(fun, sol.x)["ok"] and residual_check(scaled, sol.x)["ok"]


def test_residual_check_scale_judges_a_depleted_unknown_as_newton_does():
    # gallery learnings row 115: a component far below its field's size
    b = np.array([1.0, 1e-50])

    def fun(x):
        return x**2 - b, diags_array(2.0 * x, format="csc")

    sol = newton(fun, np.ones(2))
    assert sol.success and sol.x[1] > 1e-12          # accepted against the field's size
    plain = residual_check(fun, sol.x)
    assert not plain["ok"] and plain["backward_error"] > 0.1
    field = residual_check(fun, sol.x, scale="field")
    assert field["ok"] and field["backward_error"] <= sol.backward_error
    assert residual_check(fun, sol.x, scale=1.0)["backward_error"] == field["backward_error"]
    assert field["max_abs_residual"] == plain["max_abs_residual"]


def test_residual_check_field_scale_is_per_field_and_validated():
    x = np.array([[1.0, 1e-3], [1e-30, 2e-3]])     # two fields along the last axis

    def fun(u):
        return u.ravel() - x.ravel() + np.array([0.0, 0.0, 0.0, 1e-20]), csc_array(np.eye(4))

    field = residual_check(fun, x, scale="field")["backward_error"]
    assert np.isclose(field, 1e-20 / (2e-3 + 2e-3 + 1e-20))   # judged against field 1's size, 2e-3
    per_field = residual_check(fun, x, scale=np.array([1.0, 2e-3]))["backward_error"]
    assert per_field == field
    with pytest.raises(ValueError):
        residual_check(fun, x, scale="fields")
    with pytest.raises(ValueError):
        residual_check(fun, x, scale=0.0)
    with pytest.raises(ValueError):
        residual_check(fun, x, scale=-1.0)


def test_residual_check_field_matches_newtons_backward_error():
    from pymrm.solve import _backward_error, _field_reference
    rng = np.random.default_rng(0)
    x = rng.uniform(0.5, 2.0, (5, 3))
    x[:, 2] *= 1e-9                                  # a field below the 1e-6 floor
    jac = csc_array(np.diag(rng.uniform(1.0, 3.0, 15)) + np.diag(np.ones(14), 1))
    g = rng.normal(scale=1e-14, size=15)

    def fun(u):
        return g, jac

    newton_measure = _backward_error(g, jac, x.ravel(), _field_reference(x.ravel(), x.shape, None))
    assert residual_check(fun, x, scale="field")["backward_error"] == newton_measure


def test_check_jacobian_fails_on_nan_probe_and_mixed_scales():
    x = np.array([1e-4, 1e3])

    def log_wrong(v):  # derivative wrong by a factor 2 on a small unknown
        return np.log(v), np.diag([2.0 / v[0], 1.0 / v[1]])
    assert not check_jacobian(log_wrong, x)["ok"]

    def scaled_wrong(v):  # 10 % error in the entry of the large unknown
        return np.array([v[0] ** 2, v[1] ** 2]), np.diag([2.0 * v[0], 2.2 * v[1]])
    assert not check_jacobian(scaled_wrong, x)["ok"]

    def correct(v):
        return np.array([v[0] ** 2, np.log(v[1])]), np.diag([2.0 * v[0], 1.0 / v[1]])
    assert check_jacobian(correct, x)["ok"]


def test_observed_orders_rejects_variable_ratio_and_flags_divergence():
    import pytest
    with pytest.raises(ValueError):
        observed_orders(lambda n: 1.0 / n**2, (10, 20, 30))
    out = observed_orders(lambda n: float(n % 3), (10, 20, 40))
    assert np.isnan(out["error_estimate"]) or out["orders"][-1] > 0


def test_checks_reachable_after_bare_import():
    # the package docstring and the plugin call pymrm.checks.residual_check after `import pymrm`
    import pymrm
    assert pymrm.checks.residual_check is residual_check
