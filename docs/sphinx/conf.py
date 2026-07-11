# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from pathlib import Path
from sphinx.ext.apidoc import main as apidoc_main
sys.path.insert(0, os.path.abspath('../..'))    # this path should point at the root directory
sys.path.insert(0, str(Path('../..', 'src').resolve()))  # patch so we don't need to install the repository for sphinx to work

try:
    from src.pymrm._version import __version__      # version file created by setuptools-scm
except ImportError:
    __version__ = "unknown (this can happen if you run sphinx without installing the package first)"

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
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


def run_apidoc(_):
    docs_dir = Path(__file__).resolve().parent
    package_dir = docs_dir.parent.parent / 'src' / 'pymrm'
    apidoc_main(['-f', '-o', str(docs_dir), str(package_dir)])
    package_rst = docs_dir / 'pymrm.rst'
    if package_rst.exists():
        content = package_rst.read_text(encoding='utf-8')
        marker = "\nModule contents\n---------------\n\n.. automodule:: pymrm\n"
        if marker in content:
            content = content.split(marker, 1)[0].rstrip() + "\n"
            package_rst.write_text(content, encoding='utf-8')


def setup(app):
    app.connect('builder-inited', run_apidoc)
