"""
pymrm: A Python Package for Multiphase Reactor Modeling

This package provides a comprehensive set of tools for modeling multiphase reactors,
including grid generation, numerical operators, convection schemes, interpolation methods,
nonlinear solvers, and utility functions.

Submodules:
- grid: Functions for generating uniform and non-uniform grids.
- operator: Construction of gradient and divergence operators for finite volume methods.
- convection: High-resolution convection schemes and TVD limiters.
- interpolate: Interpolation techniques between staggered and cell-centered grids.
- solve: Nonlinear solvers and numerical approaches.
- helpers: Utility functions supporting core operations.
- numjac: Numerical Jacobian construction for nonlinear systems.

Example Usage:
.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from pymrm import construct_grad, construct_div

    # Define the grid
    shape = (100,)
    x_f = np.linspace(0, 1, shape[0]+1)
    x_c = 0.5*(x_f[1:] + x_f[:-1])

    # Set boundary conditions
    bc_L = {a: 0, b:1, d:1}
    bc_R = {a: 0, b:1, d:0}
    grad_mat, grad_bc = construct_grad(shape, x_f, x_c, bc=(bc_L, bc_R))
    div_mat = construct_div(shape, x_f)
    lapl_mat = div_mat @ grad_mat
    lapl_bc = div_mat @ grad_bc

    c = np.zeros(shape)
    c[:] = lapl_mat.solve(lapl_bc)
    plt.plot(x_c, c)

Authors:
- E.A.J.F. Peters
- M. van Sint Annaland
- M. Galanti
- D.R. Rieder

License: MIT License
"""

from .grid import generate_grid, non_uniform_grid
from .operators import (
    construct_grad,
    construct_grad_int,
    construct_grad_bc,
    construct_div,
)
from .convect import (
    construct_convflux_upwind,
    construct_convflux_upwind_int,
    construct_convflux_bc,
    upwind,
    minmod,
    osher,
    clam,
    muscl,
    smart,
    stoic,
    vanleer,
)
from .interpolate import (
    interp_stagg_to_cntr,
    interp_cntr_to_stagg,
    interp_cntr_to_stagg_tvd,
    create_staggered_array,
    compute_boundary_values,
    construct_boundary_value_matrices,
)
from .solve import newton, clip_approach
from .numjac import NumJac, stencil_block_diagonals
from .coupling import (
    update_csc_array_indices,
    update_csr_array_indices,
    update_array_indices,
    translate_indices_to_larger_array,
    construct_interface_matrices,
)
from .helpers import construct_coefficient_matrix
from ._version import __version__

__all__ = [
    "generate_grid",
    "non_uniform_grid",
    "construct_grad",
    "construct_grad_int",
    "construct_grad_bc",
    "construct_div",
    "construct_convflux_upwind",
    "construct_convflux_upwind_int",
    "construct_convflux_bc",
    "upwind",
    "minmod",
    "osher",
    "clam",
    "muscl",
    "smart",
    "stoic",
    "vanleer",
    "interp_stagg_to_cntr",
    "interp_cntr_to_stagg",
    "interp_cntr_to_stagg_tvd",
    "create_staggered_array",
    "compute_boundary_values",
    "construct_boundary_value_matrices",
    "newton",
    "clip_approach",
    "update_csc_array_indices",
    "update_csr_array_indices",
    "update_array_indices",
    "translate_indices_to_larger_array",
    "construct_interface_matrices",
    "NumJac",
    "stencil_block_diagonals",
    "construct_coefficient_matrix",
    "__version__",
]
