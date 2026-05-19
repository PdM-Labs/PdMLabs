# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = 'PdMLabs'
copyright = '2026, Anastasios Papadopoulos, Apostolos Giannoulidis, DataLab AUTh'
author = 'Anastasios Papadopoulos, Apostolos Giannoulidis, DataLab AUTh'
release = '0.0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',  # For Google-style docstrings
    'sphinx_design',
    'sphinx_copybutton', # For copy button in code blocks
]



# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

numfig = True

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "html_admonition",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
    "linkify",
    "substitution",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

autosummary_generate = True
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = "sphinx_book_theme"
html_logo = "../PdMLabs_logo.png"
html_title = "PdMLabs Documentation"
html_copy_source = True
html_favicon = "../PdMLabs_logo.png"
html_last_updated_fmt = ""

html_theme_options = {
    "repository_url": "https://github.com/PdM-Labs/PdMLabs",
    "use_repository_button": True,
    # for more pygment styles, see: https://pygments.org/styles/
    "pygments_light_style": "tango",
    "pygments_dark_style": "lightbulb",
}

# Add custom CSS
html_css_files = [
    'styles.css',
]

html_static_path = ['_static']

# This allows us to use substitutions in the documentation
rst_prolog = """
.. include:: /_templates/substitutions.rst
"""


