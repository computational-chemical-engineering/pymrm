## How to generate documentation with Sphinx

# Quickstart
The easiest way to generate the documentation is by changing into the `sphinx` directory and to use `make` for generating the file
```bash
# here we are in the root folder of the package
cd docs/sphinx
make html
```
The documentation can then be found in the folder `docs/sphinx/_build/html/index.html

# From scratch
For setting up sphinx and generating the repository, you can do the following inside the directory where you want to generate the documentation:
```bash
sphinx-quickstart -p pymrm -a "E.A.J.F. Peters"  -l en --no-sep --quiet --ext-autodoc --ext-viewcode --extension=sphinx.ext.napoleon
```
Afterwards, adapt the `conf.py` file:
```python
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))    # this path should point to the root directory
from src.pymrm._version import __version__      # version file created by setuptools-scm

project = 'pymrm'
copyright = '2025, E.A.J.F. Peters'
author = 'E.A.J.F. Peters'

# The full version, including alpha/beta/rc tags
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
```
The API `*.rst` pages (`modules.rst` and `pymrm.rst`) are generated automatically by
the Sphinx build configuration (`conf.py`) during `make html`, so they should not be
manually generated or committed.

Ensure `modules` is present in the `index.rst` toctree:
```bash
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
```
Finally, you can invoke make to generate the documentation
```bash
# here we are in the root folder of the package
cd docs/sphinx
make html
```
