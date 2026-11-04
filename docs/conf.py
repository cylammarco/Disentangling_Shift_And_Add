"""Sphinx configuration for Read the Docs and local builds."""

from importlib.metadata import version as package_version

project = "SAAD"
author = "Tomer Shenar and contributors"
release = package_version("saad-disentangling")
extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3
templates_path = []
exclude_patterns = ["_build"]
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
autodoc_typehints = "none"
