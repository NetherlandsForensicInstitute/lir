# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import inspect
import re
from pathlib import Path

from lir import registry
from lir.config.base import ConfigParser, GenericConfigParser
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
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
    'sphinx_rtd_theme',
    'sphinx_jinja',
    'jupyter_sphinx',
]

templates_path = ['_templates']
exclude_patterns = []

# Automatically generate stub pages for autosummary entries.
autosummary_generate = True

# Napoleon settings
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
    """:return: the URI that points to the documentation of the named class or `ConfigParser` object."""
    if isinstance(class_name, ConfigParser):
        class_name = class_name.reference()

    parts = class_name.split('.')
    for i in range(1, len(parts)):
        module_name = '.'.join(parts[:-i])
        if is_module(module_name):
            return f'api/{module_name}.html#{class_name}'

    return 'api/lir.html'


def get_registry_link(registry_name: str) -> str:
    """
    Apply the full-data-fitted LR system to the case data and store the resulting LLRs as CSV.

    Parameters
    ----------
    registry_name : str
        The name of the registry entry.

    Returns
    -------
    str
        An RST link that points to the API documentation of the registry item.
    """
    real_name = registry.get(registry_name, default_config_parser=GenericConfigParser).reference()
    obj = _get_attribute_by_name(real_name)

    if inspect.ismodule(obj):
        return f':mod:`{registry_name} <{real_name}>`'
    elif inspect.isclass(obj):
        return f':class:`{registry_name} <{real_name}>`'
    elif inspect.isfunction(obj):
        return f':meth:`{registry_name} <{real_name}>`'
    else:
        raise ValueError(f'Unknown category for name {real_name}')


def get_docstr_short(class_name: str | ConfigParser) -> str:
    """:return: a short version of the docstr of the named class or `ConfigParser` object."""
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
    'apidocs_uri': get_apidocs_uri,
    'registry_link': get_registry_link,
    'docstr_short': get_docstr_short,
}
