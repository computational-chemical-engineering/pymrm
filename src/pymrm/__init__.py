"""Top-level package for :mod:`pymrm`.

The package provides numerical building blocks for multiphase reactor models,
including:

* grid generation utilities;
* sparse gradient, divergence, and convective-flux operators;
* interpolation routines between cell-centered and staggered layouts;
* numerical Jacobian approximation tools;
* nonlinear solver helpers for implicit schemes;
* coupling helpers for multi-domain/interface formulations; and
* immersed-boundary operators with SDF-based domain segmentation and a
  particle front end (exact per-particle walls/normals, contact policies).
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
from .ibm import (
    IBM,
    construct_ibm,
    apply_ibm,
    apply_ibm_vector,
    reconstruct_ghost_values,
    fill_ghost_values,
)
from .ibm_recon import (
    IBMNormalDerivative,
    construct_ibm_normal_derivative,
    construct_ibm_normal_derivative_ops,
    interface_normals,
    gfd_normal_derivative_weights,
)
from .ibm_coupling import (
    construct_ibm_interface_values,
    apply_ibm_interface,
    construct_ibm_boundary_values,
)
from .segmentation import (
    Segmentation,
    segment_domain,
    crossing_segments,
    segment_values,
    wall_contact,
    wall_patch,
    wall_values,
    combine_interface_conditions,
    segment_field,
)
from .particles import (
    Particle,
    Sphere,
    Circle,
    Box,
    AnalyticParticle,
    GridParticle,
    ParticleIBMInfo,
    construct_ibm_particles,
    contact_conditions,
)
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
    "IBM",
    "construct_ibm",
    "apply_ibm",
    "apply_ibm_vector",
    "reconstruct_ghost_values",
    "fill_ghost_values",
    "IBMNormalDerivative",
    "construct_ibm_normal_derivative",
    "construct_ibm_normal_derivative_ops",
    "interface_normals",
    "gfd_normal_derivative_weights",
    "construct_ibm_interface_values",
    "apply_ibm_interface",
    "construct_ibm_boundary_values",
    "Segmentation",
    "segment_domain",
    "crossing_segments",
    "segment_values",
    "wall_contact",
    "wall_patch",
    "wall_values",
    "combine_interface_conditions",
    "segment_field",
    "Particle",
    "Sphere",
    "Circle",
    "Box",
    "AnalyticParticle",
    "GridParticle",
    "ParticleIBMInfo",
    "construct_ibm_particles",
    "contact_conditions",
    "__version__",
]
