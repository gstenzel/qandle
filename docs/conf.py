# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("./.."))
sys.path.insert(0, os.path.abspath("./../src"))
# sys.path.insert(0, os.path.abspath("./../.."))

import myst_nb  # type: ignore # noqa: F401
import sphinx_rtd_theme  # type: ignore # noqa: F401

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "qandle"
copyright = "2024, Gerhard Stenzel"
author = "Gerhard Stenzel"
release = "0.0.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "sphinx.ext.duration",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = ["**test**", "**/jupyter_execute/**"]

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/qandle_logo.jpg"

html_sidebars = {"**": ["globaltoc.html", "relations.html", "sourcelink.html", "searchbox.html"]}
nb_execution_excludepatterns = "**test**", "**build**", "**/jupyter_execute/**"
