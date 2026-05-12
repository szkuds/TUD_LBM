# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#

# -- Path app_setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# Project root and src/ layout
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(
    0, os.path.join(PROJECT_ROOT, "tud_lbm")
)  # so `import src...` works if still used

# -- Project information -----------------------------------------------------

project = "tud_lbm"
copyright = "2025-12-10 15:57:39, Sacha Szkudlarek"
author = "Sacha Szkudlarek"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = "0.2.0"
# The full version, including alpha/beta/rc tags.
release = version

# -- General configuration ------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "autoapi.extension",
    "myst_parser",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
doctest_test_doctest_blocks = ""
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Use autoapi.extension to generate API docs -----------------

autoapi_dirs = ["../tud_lbm"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    # NOT "imported-members" — that's what causes the duplicates
]

# AutoAPI renders NamedTuple / dataclass fields both in the class summary
# table *and* as individual attribute entries, producing harmless
# "duplicate object description" warnings.  Suppress them.
suppress_warnings = ["autoapi.python_import_resolution", "duplicate_object"]


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# -- Options for Intersphinx

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    # Commonly used libraries, uncomment when used in package
    # 'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    # 'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
    # 'scikit-learn': ('https://scikit-learn.org/stable/', None),
    # 'matplotlib': ('https://matplotlib.org/stable/', None),
    # 'pandas': ('http://pandas.pydata.org/docs/', None),
}
