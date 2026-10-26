"""Sphinx configuration for the cointf documentation.

Build from the repository root with:

    pip install -e ".[docs]"
    sphinx-build -b html docs docs/_build/html
"""

import sys
from pathlib import Path

# Document the package in this checkout, even without an install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cointf  # noqa: E402

project = "cointf"
author = "Arjun Biddanda"
copyright = "2026, Arjun Biddanda"
release = cointf.__version__

extensions = [
    "myst_nb",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

# Markdown pages: $...$ math. In-page links use explicit (label)= targets;
# automatic heading anchors would also label "Notes" in every module docstring.
myst_enable_extensions = ["dollarmath", "amsmath"]

# The example notebook is committed with its outputs; do not re-execute it.
nb_execution_mode = "off"

# API reference: members listed in each module's __all__, plus kernel calls.
autodoc_default_options = {"members": True, "special-members": "__call__"}
autodoc_member_order = "bysource"
numpydoc_show_class_members = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

exclude_patterns = ["_build", "api/make_r_reference.R"]

html_theme = "furo"
html_title = "cointf"
