"""Sphinx configuration for HNEP documentation."""

from __future__ import annotations

import os
import sys
from datetime import datetime

# Make `hnep` importable for autodoc
sys.path.insert(0, os.path.abspath(".."))


# ── Project ─────────────────────────────────────────────────────────
project = "HNEP"
author = "Pratik Priyanshu"
copyright = f"{datetime.now().year}, {author}"

try:
    import hnep
    release = hnep.__version__
except Exception:  # pragma: no cover
    release = "0.1.0.dev0"
version = release


# ── Extensions ──────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",   # NumPy-style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",            # render Markdown sources too
]

myst_enable_extensions = ["colon_fence", "deflist"]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autoclass_content = "both"
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy":  ("https://numpy.org/doc/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}


# ── HTML output ─────────────────────────────────────────────────────
templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = f"HNEP {release}"
html_short_title = "HNEP"
html_static_path = ["_static"] if os.path.isdir("_static") else []

html_theme_options = {
    "sidebar_hide_name": False,
}

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
