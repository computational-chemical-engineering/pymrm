"""
Integration tests for pymrm.

These tests verify that multiple modules work together correctly by solving
physically meaningful problems with known analytical solutions.
"""

import pytest
import numpy as np
from scipy.sparse.linalg import spsolve

from pymrm import (
    generate_grid,
    construct_grad,
    construct_div,
    construct_convflux_upwind,
    newton,
    NumJac,
)
from pymrm.convect import upwind, minmod
from pymrm.interpolate import interp_cntr_to_stagg_tvd


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _make_1d_grid(n, x_left=0.0, x_right=1.0):
    shape = (n,)
    x_f = np.linspace(x_left, x_right, n + 1)
    _, x_c = generate_grid(n, x_f, generate_x_c=True)
    return shape, x_f, x_c


# ---------------------------------------------------------------------------
# 1. 1D steady-state diffusion  (d²c/dx² = 0)
# ---------------------------------------------------------------------------

class TestSteadyDiffusion1D:
    """Solve -D * d(dc/dx)/dx = 0 with Dirichlet BCs.

    Analytical solution: c(x) = c_L + (c_R - c_L) * x.
    """

    def test_linear_profile_exact(self):
        n = 20
        D = 1.0
        c_L, c_R = 0.0, 1.0
        shape, x_f, x_c = _make_1d_grid(n)

        bc = ({"a": 0, "b": 1, "d": c_L}, {"a": 0, "b": 1, "d": c_R})
        grad, grad_bc = construct_grad(shape, x_f, bc=bc)
        div = construct_div(shape, x_f)

        A = D * (div @ grad)
        b = -(D * (div @ grad_bc)).toarray().ravel()

        c = spsolve(A, b)
        c_exact = c_L + (c_R - c_L) * x_c
        np.testing.assert_allclose(c, c_exact, atol=1e-10)

    def test_non_zero_left_boundary(self):
        n = 16
        D = 2.5
        c_L, c_R = 3.0, 7.0
        shape, x_f, x_c = _make_1d_grid(n)

        bc = ({"a": 0, "b": 1, "d": c_L}, {"a": 0, "b": 1, "d": c_R})
        grad, grad_bc = construct_grad(shape, x_f, bc=bc)
        div = construct_div(shape, x_f)

        A = D * (div @ grad)
        b = -(D * (div @ grad_bc)).toarray().ravel()

        c = spsolve(A, b)
        c_exact = c_L + (c_R - c_L) * x_c
        np.testing.assert_allclose(c, c_exact, atol=1e-10)

    def test_neumann_bc_right(self):
        """Left Dirichlet, right homogeneous Neumann (zero flux)."""
        n = 10
        D = 1.0
        c_L = 2.0
        shape, x_f, x_c = _make_1d_grid(n)

        # Right BC: a=1, b=0, d=0 → dc/dn = 0 (Neumann)
        bc = ({"a": 0, "b": 1, "d": c_L}, {"a": 1, "b": 0, "d": 0.0})
        grad, grad_bc = construct_grad(shape, x_f, bc=bc)
        div = construct_div(shape, x_f)

        A = D * (div @ grad)
        b = -(D * (div @ grad_bc)).toarray().ravel()

        c = spsolve(A, b)
        # With Neumann on right: solution is constant c = c_L
        np.testing.assert_allclose(c, c_L, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. 1D steady-state convection-diffusion  (v * dc/dx - D * d²c/dx² = 0)
# ---------------------------------------------------------------------------

class TestSteadyConvectionDiffusion1D:
    """Solve v * dc/dx - D * d²c/dx² = 0 with Dirichlet BCs.

    Analytical solution: c = (exp(Pe*x) - 1) / (exp(Pe) - 1)
    where Pe = v * L / D.
    """

    def _solve_conv_diff(self, n, Pe):
        D = 1.0
        v_val = Pe * D  # so that Pe = v/D over L=1
        shape, x_f, x_c = _make_1d_grid(n)
        bc = ({"a": 0, "b": 1, "d": 0.0}, {"a": 0, "b": 1, "d": 1.0})
        grad, grad_bc = construct_grad(shape, x_f, bc=bc)
        div = construct_div(shape, x_f)
        conv, conv_bc = construct_convflux_upwind(shape, x_f, bc=bc, v=v_val)

        # Steady: div @ (v*c - D*grad*c) = 0
        A = div @ conv - D * (div @ grad)
        b = -(div @ conv_bc - D * (div @ grad_bc)).toarray().ravel()
        return spsolve(A, b), x_c

    def test_low_peclet(self):
        """Low Peclet number: upwind scheme approaches exact solution."""
        n = 100
        Pe = 1.0
        c, x_c = self._solve_conv_diff(n, Pe)
        c_exact = (np.exp(Pe * x_c) - 1) / (np.exp(Pe) - 1)
        # Upwind is first-order; allow 5% relative tolerance
        np.testing.assert_allclose(c, c_exact, rtol=0.05)

    def test_zero_velocity_gives_linear(self):
        """Zero velocity: reduces to pure diffusion → linear profile."""
        n = 20
        shape, x_f, x_c = _make_1d_grid(n)
        bc = ({"a": 0, "b": 1, "d": 0.0}, {"a": 0, "b": 1, "d": 1.0})
        grad, grad_bc = construct_grad(shape, x_f, bc=bc)
        div = construct_div(shape, x_f)
        conv, conv_bc = construct_convflux_upwind(shape, x_f, bc=bc, v=0.0)
        D = 1.0
        A = div @ conv - D * (div @ grad)
        b = -(div @ conv_bc - D * (div @ grad_bc)).toarray().ravel()
        c = spsolve(A, b)
        np.testing.assert_allclose(c, x_c, atol=1e-9)


# ---------------------------------------------------------------------------
# 3. 1D nonlinear reaction-diffusion  (D * d²c/dx² - k*c = 0)
# ---------------------------------------------------------------------------

class TestNonlinearReactionDiffusion1D:
    """Solve D * d²c/dx² - k*c = 0 with Dirichlet BCs via Newton's method.

    Analytical solution:
        c(x) = c_L * sinh(phi*(L-x)/L) / sinh(phi)
                + c_R * sinh(phi*x/L) / sinh(phi)
    where phi = L * sqrt(k/D) is the Thiele modulus.
    """

    def test_thiele_modulus(self):
        n = 80
        D = 1.0
        k = 9.0       # phi = sqrt(k/D) = 3
        c_L = 1.0
        c_R = 1.0
        L = 1.0

        shape, x_f, x_c = _make_1d_grid(n, 0.0, L)
        bc = ({"a": 0, "b": 1, "d": c_L}, {"a": 0, "b": 1, "d": c_R})
        grad, grad_bc = construct_grad(shape, x_f, bc=bc)
        div = construct_div(shape, x_f)

        # Laplacian operator and BC constant term
        # sign convention: A_lap @ c + b_bc = D * div @ (grad @ c + grad_bc)
        A_lap = D * (div @ grad)
        b_bc = (D * (div @ grad_bc)).toarray().ravel()

        numjac = NumJac(shape)

        # Residual: D*d2c/dx2 - k*c = 0
        # => (A_lap - k*I) @ c + b_bc = 0
        def residual(c):
            return A_lap @ c - k * c + b_bc

        def f_and_jac(c):
            g = residual(c)
            _, jac = numjac(residual, c, f_value=g)
            return g, jac

        x0 = np.ones(n)
        sol = newton(f_and_jac, x0)
        assert sol.success

        phi = L * np.sqrt(k / D)
        c_exact = (
            c_L * np.sinh(phi * (L - x_c) / L) / np.sinh(phi)
            + c_R * np.sinh(phi * x_c / L) / np.sinh(phi)
        )
        # FVM second-order accuracy: O(dx^2) = O((1/80)^2) ≈ 1.6e-4
        np.testing.assert_allclose(sol.x, c_exact, rtol=1e-3)


# ---------------------------------------------------------------------------
# 4. 2D steady-state diffusion  (d²c/dx² + d²c/dy² = 0)
# ---------------------------------------------------------------------------

class TestSteadyDiffusion2D:
    """Solve the 2D Laplace equation with Dirichlet BCs in x, Neumann in y.

    BCs:
      c(0, y) = 0, c(1, y) = 1  (Dirichlet)
      dc/dy(x, 0) = dc/dy(x, 1) = 0  (Neumann)

    Analytical solution: c(x, y) = x  (uniform in y).
    """

    def test_2d_dirichlet_neumann(self):
        n_x, n_y = 10, 8
        D = 1.0
        shape = (n_x, n_y)
        x_f = np.linspace(0, 1, n_x + 1)
        y_f = np.linspace(0, 1, n_y + 1)

        # x-direction: Dirichlet c(0)=0, c(1)=1
        bc_x = ({"a": 0, "b": 1, "d": 0.0}, {"a": 0, "b": 1, "d": 1.0})
        grad_x, grad_bc_x = construct_grad(shape, x_f, bc=bc_x, axis=0)
        div_x = construct_div(shape, x_f, axis=0)

        # y-direction: homogeneous Neumann (no flux)
        bc_y = ({"a": 1, "b": 0, "d": 0.0}, {"a": 1, "b": 0, "d": 0.0})
        grad_y, grad_bc_y = construct_grad(shape, y_f, bc=bc_y, axis=1)
        div_y = construct_div(shape, y_f, axis=1)

        # 2D Laplacian
        A = D * (div_x @ grad_x + div_y @ grad_y)
        b = -(D * (div_x @ grad_bc_x + div_y @ grad_bc_y)).toarray().ravel()

        c = spsolve(A, b)

        # Analytical solution: c = x (independent of y)
        _, x_c = generate_grid(n_x, x_f, generate_x_c=True)
        c_exact = np.tile(x_c.reshape(-1, 1), (1, n_y)).ravel()
        np.testing.assert_allclose(c, c_exact, atol=1e-10)


# ---------------------------------------------------------------------------
# 5. TVD interpolation in a convection step
# ---------------------------------------------------------------------------

class TestTVDInterpolation:
    """Verify TVD interpolation schemes produce bounded, shape-correct results."""

    @pytest.mark.parametrize("limiter", [upwind, minmod])
    def test_tvd_boundedness(self, limiter):
        """TVD-interpolated face values should lie within [min(c), max(c)]."""
        n = 20
        x_f = np.linspace(0, 1, n + 1)
        x_c = 0.5 * (x_f[:-1] + x_f[1:])
        # Smooth but non-trivial profile
        c = np.sin(2 * np.pi * x_c)
        bc = ({"a": 0, "b": 1, "d": c[0]}, {"a": 0, "b": 1, "d": c[-1]})
        v = 1.0
        c_face, _ = interp_cntr_to_stagg_tvd(c, x_f, x_c, bc, v, limiter)
        # Face values should be within the cell-value range (TVD property)
        assert np.all(c_face >= c.min() - 1e-12)
        assert np.all(c_face <= c.max() + 1e-12)


# ---------------------------------------------------------------------------
# 6. Newton solver with NumJac on a multi-cell system
# ---------------------------------------------------------------------------

class TestNewtonNumJacIntegration:
    """Test that Newton + NumJac solve a nonlinear system correctly."""

    def test_quadratic_system(self):
        """Solve the system x_i^2 = i+1 for i=0..n-1."""
        n = 6
        targets = np.arange(1.0, n + 1)
        numjac = NumJac((n,))

        def residual(x):
            return x**2 - targets

        def f_and_jac(x):
            g = residual(x)
            _, jac = numjac(residual, x, f_value=g)
            return g, jac

        x0 = np.ones(n) * 2.0
        sol = newton(f_and_jac, x0)
        assert sol.success
        np.testing.assert_allclose(sol.x, np.sqrt(targets), rtol=1e-6)
