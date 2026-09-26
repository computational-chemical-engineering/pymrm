# pymrm: Python Package for Multiphase Reactor Modeling

[![CI](https://github.com/computational-chemical-engineering/pymrm/actions/workflows/test.yml/badge.svg)](https://github.com/computational-chemical-engineering/pymrm/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/pymrm.svg)](https://pypi.org/project/pymrm/)
[![Python](https://img.shields.io/pypi/pyversions/pymrm.svg)](https://pypi.org/project/pymrm/)
[![License](https://img.shields.io/github/license/computational-chemical-engineering/pymrm)](https://github.com/computational-chemical-engineering/pymrm/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/latestdoi/901029258.svg)](https://zenodo.org/badge/latestdoi/901029258)


## Overview

pymrm is a Python package for modeling multiphase reactors, developed as part of the Multiphase Reactor Modeling course at Eindhoven University of Technology. Originally implemented in Matlab, pymrm is now a Python-based, open-access tool designed for accessibility and advanced modeling.

## Features

- Tools for modeling diffusion, convection, reaction, and mass transfer.
- User-friendly interface for students and professionals.
- Extensive documentation and example-based learning.

## Installation

Install pymrm directly from PyPI:
```sh
pip install pymrm
```

For detailed instructions, see the [Installation Guide](https://github.com/computational-chemical-engineering/pymrm/blob/main/docs/installation.md).

## Quickstart

Start with the tutorials in the [`tutorials`](https://github.com/computational-chemical-engineering/pymrm/tree/main/tutorials) folder to learn the basics of pymrm. Then, explore the [`examples`](https://github.com/computational-chemical-engineering/pymrm/tree/main/examples) folder for more advanced use cases, including modeling diffusion, convection, and reaction processes.

For extensive documentation, exercises, and additional resources, visit the [PyMRM documentation site](https://computational-chemical-engineering.github.io/pymrm-book).

## Conventions for users and coding agents

These are the conventions most often got wrong, by people and by coding agents.
The same list is in the package docstring (`help(pymrm)`).

- **Boundary conditions** are `(lower, upper)` dictionaries `{"a", "b", "d"}`
  meaning `a * dc/dn + b * c = d` with `n` the **outward** normal, so at the
  lower end `dc/dn = -dc/dx`. `describe_bc(bc, x_f)` prints what a pair means.
  `{"outflow": True}` marks a pure-outflow boundary (a stirred volume's exit).
- **Shapes** keep a trailing field axis: one field on `n` cells is `(n, 1)`.
  `NumJac` couples the last axis in full, so a bare `(n,)` builds a dense Jacobian.
- **Recipe:** assemble the constant operators once, return `(g, jac)` from a
  residual function, use `NumJac` only for local terms such as reactions, and
  solve with `newton`.
- **`newton`** stops on an absolute step: scale unknowns to order one or pass
  `tol=0, rtol=...`; check the result with `pymrm.checks.residual_check`.
- **Boundary values** come from `compute_boundary_values` (its gradients are
  along the axis), never from the last cell centre.
- **`shapes_d`:** a dictionary's `d` is a coefficient on the external vector of
  boundary values; use `d = 1`.
- **Checks:** `pymrm.checks` has `check_jacobian`, `observed_orders`,
  `find_roots` and `residual_check`.

A minimal steady model, diffusion with a second-order reaction in a slab:

```python
import numpy as np
from pymrm import NumJac, construct_div, construct_grad, newton

n = 100
x_f = np.linspace(0.0, 1.0, n + 1)                   # face positions
shape = (n, 1)                                        # one field: keep a field axis
bc = ({"a": 1.0, "b": 0.0, "d": 0.0},               # x = 0: dc/dx = 0 (symmetry)
      {"a": 0.0, "b": 1.0, "d": 1.0})               # x = 1: c = 1
grad, grad_bc = construct_grad(shape, x_f, bc=bc)
div = construct_div(shape, x_f, nu=0)                # nu: 0 slab, 1 cylinder, 2 sphere
jac_const = (div @ -grad).tocsc()                     # diffusion, assembled once
g_const = (div @ -grad_bc).toarray().reshape(-1, 1)
numjac = NumJac(shape)                                # Jacobian of the local reaction


def residual(c):
    g_rxn, jac_rxn = numjac(lambda u: 4.0 * u**2, c.reshape(shape))
    return g_const + jac_const @ c.reshape(-1, 1) + g_rxn.reshape(-1, 1), jac_const + jac_rxn


result = newton(residual, np.ones(shape))
c = result.x.reshape(shape)
```

Coding agents: the pymrm repository also ships skills for building and checking
models (`plugins/pymrm/`, see its README).

## Helper utilities

Small building blocks used throughout the package are documented in
[`docs/helpers.md`](docs/helpers.md). These utilities assist with
handling boundary-condition coefficients and constructing sparse
coefficient matrices.

## Contributing

Contributions are welcome! For suggestions or enhancements, please contact us or submit a pull request.
