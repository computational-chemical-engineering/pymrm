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

Conventions (the ones most often got wrong, by people and coding agents):

* Boundary conditions are ``(lower, upper)`` dictionaries ``{"a", "b", "d"}``
  meaning ``a * dc/dn + b * c = d`` with ``n`` the OUTWARD normal, so at the
  lower end ``dc/dn = -dc/dx``. ``describe_bc(bc, x_f)`` prints what a pair
  means. ``{"outflow": True}`` marks a pure-outflow boundary (a stirred
  volume's exit).
* Keep a trailing field axis: one field on ``n`` cells has shape ``(n, 1)``.
  ``NumJac`` couples the LAST axis in full, so a bare ``(n,)`` shape builds a
  dense Jacobian.
* A model assembles its constant operators once (``construct_grad``,
  ``construct_div``, ``construct_convflux_upwind``), returns ``(g, jac)`` from a
  residual function, uses ``NumJac`` for local nonlinear terms such as
  reactions, and solves with ``newton``; the Jacobian is the sum of the
  constant operator part and the local part.
* ``newton`` stops on an ABSOLUTE step (``tol``): scale unknowns to order one or
  pass ``tol=0, rtol=...``, and check the result with
  ``pymrm.checks.residual_check``.
* Read boundary values with ``compute_boundary_values`` (its gradients are
  along the axis, not the outward normal), never from the last cell centre.
* With ``shapes_d`` a dictionary's ``d`` is a coefficient on the external
  vector of boundary values: use ``d = 1`` and pass the values in the vector.
* Where the diffusivity jumps between cells, use the harmonic mean at the face.

``pymrm.checks`` provides ``check_jacobian``, ``observed_orders``,
``find_roots`` and ``residual_check``. Tutorials and examples are not installed
with the package; see https://github.com/computational-chemical-engineering/pymrm
(folders ``tutorials`` and ``examples``) and the README for a minimal model.
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
from .helpers import construct_coefficient_matrix, describe_bc
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
    "describe_bc",
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
