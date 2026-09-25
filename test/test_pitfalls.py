"""Executable record of the pitfalls documented for modelling agents.

Each test demonstrates one entry of
``plugins/pymrm/skills/conventions/references/pitfalls.md`` (ids P1 to P8).
The tests pin CURRENT behaviour, including behaviour that is a trap rather than a
feature. If a test here fails after an API change, the change has made the
documented pitfall untrue: update ``pitfalls.md`` in the same PR, then update or
delete the test.
"""

import numpy as np
from scipy.sparse import csc_array, eye_array
from scipy.sparse.linalg import spsolve

from pymrm import (
    NumJac,
    compute_boundary_values,
    construct_coefficient_matrix,
    construct_convflux_upwind,
    construct_div,
    construct_grad,
    newton,
)


def _dense(vec):
    return np.asarray(vec.todense() if hasattr(vec, "todense") else vec).ravel()


def _uniform(n):
    x_f = np.linspace(0.0, 1.0, n + 1)
    return x_f, 0.5 * (x_f[1:] + x_f[:-1])


# P1 -------------------------------------------------------------------------
def test_p1_bc_gradient_is_along_outward_normal():
    """``a*dc/dn + b*c = d`` uses the outward normal; compute_boundary_values
    returns the gradient along +x. Flux q prescribed at the left end."""
    n, q = 50, 2.0
    x_f, x_c = _uniform(n)
    shape = (n, 1)
    bc = ({"a": 1.0, "b": 0.0, "d": q}, {"a": 0.0, "b": 1.0, "d": 0.0})
    grad, grad_bc = construct_grad(shape, x_f, x_c, bc)
    div = construct_div(shape, x_f, nu=0)
    c = spsolve((div @ -grad).tocsc(), _dense(div @ grad_bc))
    value_left, grad_left, _, _ = compute_boundary_values(c.reshape(shape), x_f, x_c, bc)
    # exact solution c = q (1 - x): dc/dx = -q, outward derivative -dc/dx = +q = d
    assert np.isclose(float(np.ravel(value_left)[0]), q)
    assert np.isclose(float(np.ravel(grad_left)[0]), -q)


# P2 -------------------------------------------------------------------------
def _composite_slab_flux_error(n, mean):
    x_f, x_c = _uniform(n)
    shape = (n, 1)
    d_cell = np.where(x_c < 0.5, 1.0, 0.1)
    if mean == "arithmetic":
        d_int = 0.5 * (d_cell[1:] + d_cell[:-1])
    else:
        d_int = 2.0 / (1.0 / d_cell[1:] + 1.0 / d_cell[:-1])
    d_face = np.concatenate([[d_cell[0]], d_int, [d_cell[-1]]]).reshape(-1, 1)
    bc = ({"a": 0.0, "b": 1.0, "d": 1.0}, {"a": 0.0, "b": 1.0, "d": 0.0})
    grad, grad_bc = construct_grad(shape, x_f, x_c, bc)
    div = construct_div(shape, x_f, nu=0)
    d_mat = construct_coefficient_matrix(d_face, shape=(n + 1, 1))
    c = spsolve((div @ (-d_mat @ grad)).tocsc(), _dense(div @ (d_mat @ grad_bc)))
    flux_in = -float(d_face[0, 0] * ((grad @ c)[0] + _dense(grad_bc)[0]))
    exact = 1.0 / (0.5 / 1.0 + 0.5 / 0.1)
    return abs(flux_in - exact) / exact


def test_p2_harmonic_face_mean_at_diffusivity_jump():
    """Arithmetic face mean converges at first order; harmonic is exact here."""
    err_a = [_composite_slab_flux_error(n, "arithmetic") for n in (10, 40, 160)]
    err_h = [_composite_slab_flux_error(n, "harmonic") for n in (10, 40, 160)]
    orders = np.log(np.array(err_a[:-1]) / np.array(err_a[1:])) / np.log(4.0)
    assert err_a[0] > 0.05                      # measured 7.2% at n=10
    assert np.allclose(orders, 1.0, atol=0.05)
    assert max(err_h) < 1e-10


# P3 -------------------------------------------------------------------------
def test_p3_numjac_bare_1d_shape_is_dense():
    """NumJac((n,)) couples every cell to every other; (n, 1) is diagonal."""
    n = 60

    def reaction(c):
        return -c**2

    _, jac_bare = NumJac((n,))(reaction, np.ones(n))
    _, jac_field = NumJac((n, 1))(reaction, np.ones((n, 1)))
    assert jac_bare.nnz == n * n
    assert jac_field.nnz == n


# P4 -------------------------------------------------------------------------
def test_p4_axes_diagonals_on_1d_shape_gives_wrong_jacobian():
    """axes_diagonals=[0] on a 1-D shape misplaces entries; the true Jacobian
    of a pointwise -c**2 at c=1 is -2 on the diagonal."""
    n = 8

    def reaction(c):
        return -c**2

    _, jac = NumJac((n,), axes_diagonals=[0])(reaction, np.ones(n))
    assert not np.allclose(jac.toarray(), -2.0 * np.eye(n), atol=1e-4)
    _, jac_ok = NumJac((n, 1))(reaction, np.ones((n, 1)))
    assert np.allclose(jac_ok.toarray(), -2.0 * np.eye(n), atol=1e-4)


# P5 -------------------------------------------------------------------------
def test_p5_shapes_d_separates_boundary_values_from_operator():
    """With shapes_d the boundary source is linear in d: assemble once, then
    multiply by a new d instead of rebuilding the operator."""
    n = 20
    x_f, x_c = _uniform(n)
    shape = (n, 1)
    bc = ({"a": 0.0, "b": 1.0, "d": 1.0}, {"a": 1.0, "b": 0.0, "d": 0.0})
    grad, grad_bc_left, _ = construct_grad(shape, x_f, x_c, bc, shapes_d=((1,), None))
    for d_new in (0.3, 2.5):
        bc_new = ({"a": 0.0, "b": 1.0, "d": d_new}, bc[1])
        grad_rebuilt, grad_bc_rebuilt = construct_grad(shape, x_f, x_c, bc_new)
        assert np.allclose((grad - grad_rebuilt).toarray(), 0.0)
        assert np.allclose(_dense(grad_bc_left @ np.array([d_new])), _dense(grad_bc_rebuilt))


# P6 -------------------------------------------------------------------------
def _tanks_outlet(n_tanks, k, outlet):
    z_f, z_c = _uniform(n_tanks)
    shape = (n_tanks, 1)
    bc_in = {"a": 0.0, "b": 1.0, "d": 1.0}
    div = construct_div(shape, z_f, nu=0)
    if outlet == "zero-gradient":
        bc = (bc_in, {"a": 1.0, "b": 0.0, "d": 0.0})
        conv, conv_bc = construct_convflux_upwind(shape, z_f, z_c, bc, v=1.0)
        mat = div @ conv
    else:
        # no flux through the exit face, outflow v*C_N added as a sink on the last cell
        bc = (bc_in, {"a": 0.0, "b": 1.0, "d": 0.0})
        conv, conv_bc = construct_convflux_upwind(shape, z_f, z_c, bc, v=1.0)
        coef = np.zeros(shape)
        coef[-1, 0] = 1.0 / (z_f[-1] - z_f[-2])
        mat = div @ conv + construct_coefficient_matrix(coef, shape=shape)
    c = spsolve((mat + k * eye_array(n_tanks)).tocsc(), -_dense(div @ conv_bc))
    return c[-1]


def test_p6_no_pure_outflow_bc():
    """The zero-gradient outlet reconstructs the exit face; for tanks in series
    that is a modelling error. The sink workaround is exact."""
    for n_tanks, err_expected in ((2, 0.0385), (8, 0.0125)):
        exact = 1.0 / (1.0 + 1.0 / n_tanks) ** n_tanks
        err_zg = _tanks_outlet(n_tanks, 1.0, "zero-gradient") / exact - 1.0
        assert abs(err_zg - err_expected) < 5e-4
        assert abs(_tanks_outlet(n_tanks, 1.0, "sink") / exact - 1.0) < 1e-12


# P7 -------------------------------------------------------------------------
def test_p7_newton_stops_on_absolute_step():
    """newton's tol is absolute on the step: a trace-level unknown reports
    success after one step and is 50x wrong. Scaling the unknown fixes it."""
    s = 1e-20

    def f(x):
        return np.array([x[0] ** 2 - s]), csc_array(np.array([[2.0 * x[0]]]))

    res = newton(f, np.array([1e-8]))
    assert res.success and res.nit == 1
    assert res.x[0] / np.sqrt(s) > 40.0

    scale = 1e-10

    def f_scaled(y):
        return (np.array([(scale * y[0]) ** 2 - s]) / s,
                csc_array(np.array([[2.0 * scale**2 * y[0] / s]])))

    res = newton(f_scaled, np.array([100.0]))
    assert res.success and np.isclose(scale * res.x[0], np.sqrt(s), rtol=1e-8)


# P8 -------------------------------------------------------------------------
def _dispersion_balance(n):
    z_f, z_c = _uniform(n)
    shape = (n, 1)
    v, d_ax, k = 1.0, 0.05, 2.0
    # Danckwerts inlet v*c - D*dc/dz = v*c_in (outward normal -z), zero-gradient outlet
    bc = ({"a": d_ax, "b": v, "d": v * 1.0}, {"a": 1.0, "b": 0.0, "d": 0.0})
    grad, grad_bc = construct_grad(shape, z_f, z_c, bc)
    conv, conv_bc = construct_convflux_upwind(shape, z_f, z_c, bc, v=v)
    div = construct_div(shape, z_f, nu=0)
    mat = div @ (conv - d_ax * grad) + k * eye_array(n)
    c = spsolve(mat.tocsc(), -_dense(div @ (conv_bc - d_ax * grad_bc)))
    _, _, value_out, _ = compute_boundary_values(c.reshape(shape), z_f, z_c, bc)
    consumed = k * np.sum(c * np.diff(z_f))
    face = v * 1.0 - v * float(np.ravel(value_out)[0]) - consumed
    last_cell = v * 1.0 - v * c[-1] - consumed
    return face, last_cell


def test_p8_outlet_value_from_the_face_not_the_last_cell():
    """The mass balance closes with the face value, not with v*C_N."""
    face, last_cell = _dispersion_balance(8)
    assert abs(face) < 1e-12
    assert abs(last_cell) > 3e-3                # measured 3.8e-3 at n=8
