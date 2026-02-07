# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import re
from pathlib import Path

from lir import registry
from lir.config.base import ConfigParser
from lir.registry import _get_attribute_by_name


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
    'autodoc2',
    'myst_parser',
    'sphinx.ext.autosummary',
    'sphinx_rtd_theme',
    'sphinx.ext.napoleon',
    'sphinx_jinja',
]

autodoc2_packages = [
    '../lir',
]

autodoc2_output_dir = 'api'

autodoc2_docstring_parser_regexes = [
    (r'.*', 'myst'),
]

autodoc2_module_all_regexes = [
    'lir.lrsystems',
    'lir.transform.pipeline',
]

suppress_warnings = [
    'toc.not_included',
]

templates_path = ['_templates']
exclude_patterns = []

# Automatically generate stub pages for autosummary entries.
autosummary_generate = True

# Napoleon settings
napoleon_google_docstring = True
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


# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 3,
}
html_sidebars = {'**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html']}


def is_module(name: str) -> bool:
    """Check whether the given name corresponds to a module in the lir package."""
    source_path = Path(__file__).parent.parent / name.replace('.', '/')
    return source_path.is_dir() or Path(f'{source_path}.py').exists()


def get_apidocs_uri(class_name: str | ConfigParser) -> str:
    """Get the URI to the API documentation page for the given class name."""
    if isinstance(class_name, ConfigParser):
        class_name = class_name.reference()

    parts = class_name.split('.')
    for i in range(1, len(parts)):
        module_name = '.'.join(parts[:-i])
        if is_module(module_name):
            return f'api/lir/{module_name}.html#{class_name}'

    return 'api/lir.html'


def get_docstr_short(class_name: str | ConfigParser) -> str:
    """Get the short form of the docstring for the given class name."""
    if isinstance(class_name, ConfigParser):
        class_name = class_name.reference()

    docstr = _get_attribute_by_name(class_name).__doc__ or ''
    docstr = re.sub('\n.*', '', docstr.strip())
    return docstr


jinja_globals = {
    'registry': registry.registry(),
}

jinja_contexts = {}

jinja_filters = {
    'apidocs_uri': lambda class_name: get_apidocs_uri(class_name),
    'docstr_short': lambda class_name: get_docstr_short(class_name),
}
