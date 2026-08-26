# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from docs import GetApidocsUri, GetDocstrShort, GetRegistryLink
from lir import registry


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LiR - Likelihood Ratio Toolkit'
copyright = '%Y, Netherlands Forensic Institute'  # noqa: A001
author = 'Netherlands Forensic Institute'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# The master toctree document.
master_doc = 'index'

pygments_style = 'sphinx'  # enable syntax highlighting

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = ['lir.']

extensions = [
    'sphinx.ext.apidoc',  # generate RST files for API documentation
    #'sphinx.ext.autodoc',
    #'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',  # include references to third party libraries
    'sphinx.ext.napoleon',  # parse NumPy and Google style docstrings
    'sphinx.ext.viewcode',  # include links to source code
    'myst_parser',
    'sphinx_rtd_theme',
    'sphinx_jinja',  # render Jinja templates in RST files
    'jupyter_sphinx',  # include Jupyter notebooks in the documentation
]

# configuration for apidoc
# see: https://www.sphinx-doc.org/en/master/usage/extensions/apidoc.html
apidoc_modules = [
    {
        'path': '../lir',
        'destination': 'api',
        'exclude_patterns': [
            '../lir/data/models.py',
            '**/base.py',
        ],
        'module_first': True,
        'separate_modules': False,
    },
]

# configuration for autosummary
# see: https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html
autosummary_generate = True

# configuration for intersphinx
# see: https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'sklearn': ('http://scikit-learn.org/stable', None),
}

# configuration for napoleon
# see: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = None

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 3,
}
html_sidebars = {'**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html']}

jinja_globals = {
    'registry': registry.registry(),
}

jinja_contexts = {}

jinja_filters = {
    'apidocs_uri': GetApidocsUri(),
    'registry_link': GetRegistryLink(),
    'docstr_short': GetDocstrShort(),
}
