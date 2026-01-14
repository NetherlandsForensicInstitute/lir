# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LiR - Likelihood Ratio Toolkit'
copyright = '%Y, Netherlands Forensic Institute'
author = 'Netherlands Forensic Institute'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# The master toctree document.
master_doc = 'index'

pygments_style = 'sphinx'       # enable syntax highlighting

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = ['lir.']

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = []

# Automatically generate stub pages for autosummary entries.
autosummary_generate = True


# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}
html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'] }
