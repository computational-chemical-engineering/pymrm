"""Pure-outflow boundary marker {"outflow": True}: face value = adjacent cell."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.sparse import eye_array
from scipy.sparse.linalg import spsolve

from pymrm import (compute_boundary_values, construct_convflux_upwind, construct_div, construct_grad,
                   interp_cntr_to_stagg_tvd, minmod)

OUT = {"outflow": True}


def _dense(m):
    return np.asarray(m.todense() if hasattr(m, "todense") else m).ravel()


def _tanks(n_tanks, k, v=1.0):
    z_f = np.linspace(0.0, 1.0, n_tanks + 1)
    shape = (n_tanks, 1)
    inlet = {"a": 0.0, "b": 1.0, "d": 1.0}
    bc = (inlet, OUT) if v > 0 else (OUT, inlet)
    conv, conv_bc = construct_convflux_upwind(shape, z_f, bc=bc, v=v)
    div = construct_div(shape, z_f, nu=0)
    mat = (div @ conv + k * eye_array(n_tanks)).tocsc()
    rhs = -_dense(div @ conv_bc)
    return mat, rhs, bc, z_f


@pytest.mark.parametrize("n_tanks", [1, 2, 6])
def test_tanks_in_series_steady_is_exact(n_tanks):
    k = 1.0
    mat, rhs, _, _ = _tanks(n_tanks, k)
    c = spsolve(mat, rhs)
    assert np.isclose(c[-1], 1.0 / (1.0 + k / n_tanks) ** n_tanks, rtol=1e-12)


def test_reversed_flow_outflow_at_lower_end():
    mat, rhs, _, _ = _tanks(4, 1.0, v=-1.0)
    c = spsolve(mat, rhs)
    assert np.isclose(c[0], 1.0 / 1.25**4, rtol=1e-12)


def test_tanks_transient_matches_ode():
    n_tanks, k, dt, t_end = 3, 0.3 * 10.0, 1e-3, 0.5
    mat, rhs, _, _ = _tanks(n_tanks, k)
    lu_mat = (mat + eye_array(n_tanks) / dt).tocsc()
    c = np.zeros(n_tanks)
    for _ in range(int(round(t_end / dt))):
        c = spsolve(lu_mat, rhs + c / dt)

    def ode(t, y):
        feed = np.concatenate([[1.0], y[:-1]])
        return n_tanks * (feed - y) - k * y
    ref = solve_ivp(ode, (0.0, t_end), np.zeros(n_tanks), rtol=1e-10, atol=1e-12).y[:, -1]
    assert np.allclose(c, ref, atol=5e-3)   # backward Euler, first order in dt


def test_boundary_values_and_gradient_at_outflow():
    mat, rhs, bc, z_f = _tanks(5, 1.0)
    c = spsolve(mat, rhs).reshape(-1, 1)
    _, _, value_out, grad_out = compute_boundary_values(c, z_f, bc=bc)
    assert np.isclose(float(np.ravel(value_out)[0]), c[-1, 0])
    assert float(np.ravel(grad_out)[0]) == 0.0
    grad, grad_bc = construct_grad(c.shape, z_f, bc=bc)
    assert np.isclose((grad @ c.ravel())[-1] + _dense(grad_bc)[-1], 0.0)


def test_tvd_face_value_and_correction_at_outflow():
    z_f = np.linspace(0.0, 1.0, 11)
    c = np.linspace(1.0, 0.1, 10).reshape(-1, 1) ** 2
    bc = ({"a": 0.0, "b": 1.0, "d": 1.0}, OUT)
    face, delta = interp_cntr_to_stagg_tvd(c, z_f, bc=bc, v=1.0, tvd_limiter=minmod)
    assert np.isclose(face[-1, 0], c[-1, 0]) and delta[-1, 0] == 0.0


def test_shapes_d_path_matches():
    z_f = np.linspace(0.0, 1.0, 5)
    shape = (4, 1)
    # with shapes_d the dictionary's d is a coefficient on the external vector: d = 1
    conv, conv_bc = construct_convflux_upwind(shape, z_f, bc=({"a": 0.0, "b": 1.0, "d": 2.0}, OUT), v=1.0)
    conv_d, conv_bc_left, _ = construct_convflux_upwind(
        shape, z_f, bc=({"a": 0.0, "b": 1.0, "d": 1.0}, OUT), v=1.0, shapes_d=((1,), None))
    assert np.allclose((conv - conv_d).toarray(), 0.0)
    assert np.allclose(_dense(conv_bc_left @ np.array([2.0])), _dense(conv_bc))


def test_outflow_marker_rejects_coefficients():
    with pytest.raises(ValueError):
        construct_convflux_upwind((3, 1), np.linspace(0, 1, 4),
                                  bc=({"a": 0, "b": 1, "d": 1}, {"outflow": True, "a": 1.0}), v=1.0)
