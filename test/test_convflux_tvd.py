"""ConvFluxTVD: TVD face fluxes with exact Jacobians."""
import numpy as np
import pytest

import pymrm
from pymrm import (
    ConvFluxTVD,
    construct_convflux_upwind,
    construct_div,
    construct_grad,
    interp_cntr_to_stagg_tvd,
    newton,
)

LIMITERS = [pymrm.upwind, pymrm.minmod, pymrm.osher, pymrm.clam, pymrm.muscl, pymrm.smart,
            pymrm.stoic, pymrm.vanleer]
BCS = {
    "dirichlet-zerogradient": ({"a": 0, "b": 1, "d": [[1.0, 0.5]]}, {"a": 1, "b": 0, "d": 0}),
    "danckwerts-outflow": ({"a": 0.3, "b": 1, "d": [[1.0, 0.5]]}, {"outflow": True}),
    "robin-robin": ({"a": 1.0, "b": 0.6, "d": [[0.2, 0.4]]}, {"a": 0.5, "b": 1.0, "d": [[0.7, 0.1]]}),
}


def _velocity(rng, shape_f, kind):
    v = rng.random(shape_f) + 0.2
    if kind == "negative":
        return -v
    if kind == "mixed":
        return v * np.sign(rng.random(shape_f) - 0.4)
    return v


def _state(rng, shape, axis):
    """A decaying profile with noise, so that most faces are limited (0 < c_norm < 1)."""
    n = shape[axis]
    profile = np.linspace(1.0, 0.2, n).reshape([n if i == axis else 1 for i in range(len(shape))])
    return profile ** (1 + rng.random(shape)) + 0.02 * rng.standard_normal(shape)


def _directional(f, x, dx, h=1e-7):
    return ((f(x + h * dx) - f(x - h * dx)) / (2 * h)).ravel()


@pytest.mark.parametrize("n", [1, 2, 3, 7])
@pytest.mark.parametrize("bc_name", list(BCS))
@pytest.mark.parametrize("kind", ["positive", "negative", "mixed"])
@pytest.mark.parametrize("axis", [0, 1])
def test_flux_and_jacobian(n, bc_name, kind, axis):
    rng = np.random.default_rng(n + 10 * axis)
    shape = (n, 3, 2) if axis == 0 else (3, n, 2)
    shape_f = tuple(s + (i == axis) for i, s in enumerate(shape))
    x_f = np.cumsum(np.r_[0.0, rng.random(n) + 0.4])        # non-uniform
    bc = BCS[bc_name]
    for limiter in LIMITERS:
        v = _velocity(rng, shape_f, kind)
        tvd = ConvFluxTVD(shape, x_f, bc=bc, tvd_limiter=limiter, axis=axis)
        c = _state(rng, shape, axis)
        flux, jac = tvd(c, v)
        face, _ = interp_cntr_to_stagg_tvd(c, x_f, None, bc, v, limiter, axis=axis)
        np.testing.assert_allclose(flux, v * face, rtol=0, atol=1e-13)
        dc = rng.standard_normal(shape)
        fd = _directional(lambda x: tvd(x, v)[0], c, dc)
        np.testing.assert_allclose(jac @ dc.ravel(), fd, rtol=0, atol=1e-6 * max(1, np.abs(fd).max()))


@pytest.mark.parametrize("n", [1, 2, 5])
def test_boundary_data_and_velocity_jacobians(n):
    rng = np.random.default_rng(n)
    shape, shape_d = (n, 3, 2), (1, 3, 2)
    x_f = np.cumsum(np.r_[0.0, rng.random(n) + 0.4])
    bc = ({"a": 0.4, "b": 1, "d": 1.0}, {"a": 0.7, "b": 0.5, "d": [[0.3, 2.0]]})
    for limiter in (pymrm.minmod, pymrm.vanleer, pymrm.smart):
        tvd = ConvFluxTVD(shape, x_f, bc=bc, tvd_limiter=limiter, shapes_d=(shape_d, shape_d))
        c, ext = _state(rng, shape, 0), [rng.random(shape_d) + 0.5, rng.random(shape_d)]
        v = _velocity(rng, (n + 1, 3, 2), "mixed")
        flux, jac, jac_left, jac_right, jac_v = tvd(c, v, d=ext, jac_v=True)
        folded = tuple({**b, "d": np.asarray(b["d"]) * e} for b, e in zip(bc, ext))
        face, _ = interp_cntr_to_stagg_tvd(c, x_f, None, folded, v, limiter, axis=0)
        np.testing.assert_allclose(flux, v * face, rtol=0, atol=1e-13)
        for i, jac_bc in enumerate((jac_left, jac_right)):
            de = rng.standard_normal(shape_d)

            def flux_of(e, i=i):
                pair = list(ext)
                pair[i] = e
                return tvd(c, v, d=pair)[0]
            np.testing.assert_allclose(jac_bc @ de.ravel(), _directional(flux_of, ext[i], de), atol=1e-6)
        dv = rng.standard_normal(v.shape)
        np.testing.assert_allclose(jac_v @ dv.ravel(), _directional(lambda w: tvd(c, w, d=ext)[0], v, dv),
                                   atol=1e-6)
        np.testing.assert_allclose(jac_v.diagonal(), face.ravel(), atol=1e-14)


def test_velocity_columns_follow_v_shape():
    shape = (6, 4, 1)
    tvd = ConvFluxTVD(shape, np.linspace(0, 1, 7), bc=({"a": 0, "b": 1, "d": 1}, {"outflow": True}))
    c = _state(np.random.default_rng(0), shape, 0)
    v = np.full((7, 1, 1), 0.8)                     # one velocity per axial face
    _, _, jac_v = tvd(c, v, jac_v=True)
    assert jac_v.shape == (7 * 4, 7)
    dv = np.random.default_rng(1).standard_normal(v.shape)
    np.testing.assert_allclose(jac_v @ dv.ravel(), _directional(lambda w: tvd(c, w)[0], v, dv), atol=1e-6)


@pytest.mark.parametrize("bc_name", list(BCS))
def test_upwind_limiter_reproduces_upwind_operator(bc_name):
    rng = np.random.default_rng(2)
    shape, shape_d = (6, 3, 2), (1, 3, 2)
    x_f = np.cumsum(np.r_[0.0, rng.random(6) + 0.4])
    v = _velocity(rng, (7, 3, 2), "mixed")
    bc = BCS[bc_name]
    matrix, matrix_bc = construct_convflux_upwind(shape, x_f, bc=bc, v=v)
    _, jac = ConvFluxTVD(shape, x_f, bc=bc, tvd_limiter=pymrm.upwind)(_state(rng, shape, 0), v)
    np.testing.assert_allclose(jac.toarray(), matrix.toarray(), atol=1e-14)
    matrix, left, right = construct_convflux_upwind(shape, x_f, bc=bc, v=v, shapes_d=(shape_d, shape_d))
    ext = [rng.random(shape_d), rng.random(shape_d)]
    _, jac, jac_left, jac_right = ConvFluxTVD(shape, x_f, bc=bc, tvd_limiter=pymrm.upwind,
                                              shapes_d=(shape_d, shape_d))(np.zeros(shape), v, d=ext)
    np.testing.assert_allclose(jac.toarray(), matrix.toarray(), atol=1e-14)
    np.testing.assert_allclose(jac_left.toarray(), left.toarray(), atol=1e-14)
    np.testing.assert_allclose(jac_right.toarray(), right.toarray(), atol=1e-14)


def test_pattern_does_not_depend_on_flow_direction():
    rng = np.random.default_rng(4)
    shape = (8, 3, 2)
    tvd = ConvFluxTVD(shape, np.linspace(0, 1, 9), bc=BCS["robin-robin"], tvd_limiter=pymrm.vanleer)
    c = _state(rng, shape, 0)
    jacs = [tvd(c, _velocity(rng, (9, 3, 2), kind))[1] for kind in ("positive", "negative", "mixed")]
    for jac in jacs[1:]:
        np.testing.assert_array_equal(jac.indptr, jacs[0].indptr)
        np.testing.assert_array_equal(jac.indices, jacs[0].indices)


@pytest.mark.parametrize("limiter", LIMITERS[1:], ids=lambda f: f.__name__)
@pytest.mark.parametrize("x_norm", [(0.5, 0.75), (1 / 3, 2 / 3), (0.42, 0.61), (0.58, 0.86)])
def test_limiter_derivative(limiter, x_norm):
    c_norm = np.linspace(-0.2, 1.2, 1401)
    x_c, x_d = (np.full_like(c_norm, x) for x in x_norm)
    h = 1e-7
    left = (limiter(c_norm, x_c, x_d) - limiter(c_norm - h, x_c, x_d)) / h
    right = (limiter(c_norm + h, x_c, x_d) - limiter(c_norm, x_c, x_d)) / h
    smooth = np.abs(left - right) < 1e-5                 # skip points next to a kink
    np.testing.assert_allclose(limiter.derivative(c_norm, x_c, x_d)[smooth], (left[smooth] + right[smooth]) / 2,
                               atol=1e-5)
    assert smooth.sum() > 1300


def test_custom_limiter_uses_numerical_derivative():
    def koren_like(c_norm, x_c, x_d):                     # a smooth limiter without .derivative
        return np.maximum(0, np.minimum(c_norm * (1 - c_norm), 1 - c_norm)) * (x_d - x_c) / (x_c * (1 - x_c))

    rng = np.random.default_rng(5)
    shape = (9, 3, 2)
    tvd = ConvFluxTVD(shape, np.linspace(0, 1, 10), bc=BCS["dirichlet-zerogradient"], tvd_limiter=koren_like)
    c, v = _state(rng, shape, 0), 1.3
    _, jac = tvd(c, v)
    dc = rng.standard_normal(shape)
    np.testing.assert_allclose(jac @ dc.ravel(), _directional(lambda x: tvd(x, v)[0], c, dc), atol=1e-6)


@pytest.mark.parametrize("limiter", LIMITERS[1:], ids=lambda f: f.__name__)
def test_steady_newton_converges(limiter):
    # u dc/dx = D d2c/dx2 - k c on [0, 1], Danckwerts inlet, outflow outlet: convection dominated
    n, u, D, k = 200, 1.0, 1e-3, 2.0
    x_f = np.linspace(0, 1, n + 1)
    bc_conv = ({"a": 0, "b": 1, "d": 1.0}, {"outflow": True})
    tvd = ConvFluxTVD((n,), x_f, bc=bc_conv, tvd_limiter=limiter)
    grad, grad_bc = construct_grad((n,), x_f, bc=({"a": 0, "b": 1, "d": 1.0}, {"a": 1, "b": 0, "d": 0}))
    div = construct_div((n,), x_f)
    diff, diff_bc = -D * (div @ grad), -D * (div @ grad_bc).toarray().ravel()

    def model(c):
        flux, jac = tvd(c, u)
        g = div @ flux + diff @ c + diff_bc + k * c
        return g, div @ jac + diff + k * np.eye(n)

    sol = newton(model, np.zeros(n), tol=1e-12, maxfev=40, damping="monotone")
    assert sol.success and sol.nit <= 25
    assert np.abs(model(sol.x)[0]).max() < 1e-10


def test_pressure_chain_rule():
    # flux = v(p) c_f with v = -K dp/dx on the faces: d flux/dp = jac_v @ (-K grad)
    n, K = 12, 0.3
    x_f = np.linspace(0, 1, n + 1)
    grad, grad_bc = construct_grad((n,), x_f, bc=({"a": 0, "b": 1, "d": 2.0}, {"a": 0, "b": 1, "d": 1.0}))
    tvd = ConvFluxTVD((n,), x_f, bc=({"a": 0, "b": 1, "d": 1.0}, {"outflow": True}), tvd_limiter=pymrm.vanleer)
    rng = np.random.default_rng(6)
    c, p = _state(rng, (n,), 0), np.linspace(2, 1, n) + 0.01 * rng.standard_normal(n)

    def flux_of(p):
        return tvd(c, -K * (grad @ p + grad_bc.toarray().ravel()))[0]
    _, _, jac_v = tvd(c, -K * (grad @ p + grad_bc.toarray().ravel()), jac_v=True)
    dp = rng.standard_normal(n)
    np.testing.assert_allclose(jac_v @ (-K * grad) @ dp, _directional(flux_of, p, dp), atol=1e-6)


def test_input_errors():
    tvd = ConvFluxTVD((4, 2), np.linspace(0, 1, 5), bc=BCS["robin-robin"], shapes_d=((1, 2), None))
    with pytest.raises(ValueError, match="pass d"):
        tvd(np.zeros((4, 2)), 1.0)
    with pytest.raises(ValueError, match="does not fit"):
        tvd(np.zeros((4, 2)), np.ones(3), d=(np.ones(2), None))


def test_velocity_accepted_as_by_upwind_constructor():
    rng = np.random.default_rng(8)
    shape = (6, 3)
    x_f = np.cumsum(np.r_[0.0, rng.random(6) + 0.4])
    bc = ({"a": 0, "b": 1, "d": 1.0}, {"outflow": True})
    c = _state(rng, shape, 0)
    tvd = ConvFluxTVD(shape, x_f, bc=bc, tvd_limiter=pymrm.vanleer)
    v_face = rng.random(7) + 0.2                     # 1-D, one value per face along axis 0
    flux, jac, jac_v = tvd(c, v_face, jac_v=True)
    flux_2d, jac_2d = tvd(c, np.broadcast_to(v_face[:, None], (7, 3)))
    np.testing.assert_allclose(flux, flux_2d, rtol=0, atol=0)
    np.testing.assert_allclose(jac.toarray(), jac_2d.toarray(), rtol=0, atol=0)
    assert jac_v.shape == (7 * 3, 7)
    dv = rng.standard_normal(7)
    np.testing.assert_allclose(jac_v @ dv, _directional(lambda w: tvd(c, w)[0], v_face, dv), atol=1e-6)
    v_cell = rng.random(6) + 0.2                     # cell-centred: interpolated, as in the upwind constructor
    matrix, _ = construct_convflux_upwind(shape, x_f, bc=bc, v=v_cell)
    _, jac = ConvFluxTVD(shape, x_f, bc=bc, tvd_limiter=pymrm.upwind)(c, v_cell)
    np.testing.assert_allclose(jac.toarray(), matrix.toarray(), atol=1e-14)
    with pytest.raises(ValueError, match="on the faces"):
        tvd(c, v_cell, jac_v=True)
    one = ConvFluxTVD((1, 3), np.array([0.0, 1.0]), bc=bc, tvd_limiter=pymrm.vanleer)
    c1, v1 = rng.random((1, 3)), rng.random(3) + 0.2  # one cell: the cell value is the face value
    _, _, jac_v = one(c1, v1, jac_v=True)
    dv = rng.standard_normal(3)
    np.testing.assert_allclose(jac_v @ dv, _directional(lambda w: one(c1, w)[0], v1, dv), atol=1e-6)


def test_csr_format_and_negative_axis():
    rng = np.random.default_rng(7)
    shape = (3, 4, 6)                                 # convection along the last axis
    x_f = np.cumsum(np.r_[0.0, rng.random(6) + 0.4])
    bc = ({"a": 0, "b": 1, "d": 1.0}, {"outflow": True})
    c = _state(rng, shape, 2)
    v = _velocity(rng, (3, 4, 7), "mixed")
    flux, jac = ConvFluxTVD(shape, x_f, bc=bc, tvd_limiter=pymrm.vanleer, axis=-1, format="csr")(c, v)
    face, _ = interp_cntr_to_stagg_tvd(c, x_f, None, bc, v, pymrm.vanleer, axis=2)
    np.testing.assert_allclose(flux, v * face, atol=1e-13)
    assert jac.format == "csr"
    _, jac_csc = ConvFluxTVD(shape, x_f, bc=bc, tvd_limiter=pymrm.vanleer, axis=2)(c, v)
    np.testing.assert_allclose(jac.toarray(), jac_csc.toarray(), atol=1e-14)


def test_returned_jacobians_are_independent():
    shape = (6, 2)
    tvd = ConvFluxTVD(shape, np.linspace(0, 1, 7), bc=({"a": 0, "b": 1, "d": 1.0}, {"outflow": True}),
                      tvd_limiter=pymrm.vanleer)
    c, v = _state(np.random.default_rng(8), shape, 0), np.where(np.arange(7)[:, None] % 3 == 0, 0.0, 1.0)
    first = tvd(c, v)[1]
    first.eliminate_zeros()                           # structural change to one returned matrix
    again = tvd(c, v)[1]
    fresh = ConvFluxTVD(shape, np.linspace(0, 1, 7), bc=({"a": 0, "b": 1, "d": 1.0}, {"outflow": True}),
                        tvd_limiter=pymrm.vanleer)(c, v)[1]
    np.testing.assert_array_equal(again.toarray(), fresh.toarray())


def test_custom_limiter_nonzero_outside_unit_interval():
    def central(c_norm, x_c, x_d):                   # an unlimited linear scheme, nonzero for any c_norm
        return (x_d - x_c) / (1 - x_c) * (1 - c_norm)

    rng = np.random.default_rng(9)
    shape = (10, 2)
    tvd = ConvFluxTVD(shape, np.linspace(0, 1, 11), bc=({"a": 0, "b": 1, "d": 1.0}, {"outflow": True}),
                      tvd_limiter=central)
    c = rng.random(shape)                             # oscillating: many c_norm outside (0, 1)
    _, jac = tvd(c, 1.0)
    dc = rng.standard_normal(shape)
    np.testing.assert_allclose(jac @ dc.ravel(), _directional(lambda x: tvd(x, 1.0)[0], c, dc), atol=1e-6)
