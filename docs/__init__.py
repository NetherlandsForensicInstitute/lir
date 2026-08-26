import inspect
import re
from pathlib import Path

from lir import registry
from lir.config.base import ConfigParser, GenericConfigParser
from lir.registry import _get_attribute_by_name


def is_module(name: str) -> bool:
    """Check whether the given name corresponds to a module in the lir package."""
    source_path = Path(__file__).parent.parent / name.replace('.', '/')
    return source_path.is_dir() or Path(f'{source_path}.py').exists()


class GetApidocsUri:
    """Function class."""

    def __call__(self, class_name: str | ConfigParser) -> str:
        """Obtain a URI that points to the documentation of the named class or `ConfigParser` object."""
        if isinstance(class_name, ConfigParser):
            class_name = class_name.reference()

        parts = class_name.split('.')
        for i in range(1, len(parts)):
            module_name = '.'.join(parts[:-i])
            if is_module(module_name):
                return f'api/{module_name}.html#{class_name}'

        return 'api/lir.html'


class GetRegistryLink:
    """Function class."""

    def __call__(self, registry_name: str) -> str:
        """
        Obtain a link in RST format to the API documentation of a registry entry.

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


class GetDocstrShort:
    """Function class."""

    def __call__(self, class_name: str | ConfigParser) -> str:
        """Obtain a brief docstr for a named class or `ConfigParser` object."""
        if isinstance(class_name, ConfigParser):
            class_name = class_name.reference()

        docstr = _get_attribute_by_name(class_name).__doc__ or ''
        docstr = re.sub('\n.*', '', docstr.strip())
        return docstr
