import importlib.resources
import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

import confidence

from lir.config.base import ConfigParser

from . import resources as package_resources


LOG = logging.getLogger(__name__)


def _get_attribute_by_name(name: str) -> Any:
    """Resolve the corresponding function or class in this project from the configuration string."""
    parts = name.split('.')

    LOG.debug(f'{name} is being resolved')

    # split the full name into a module name and a class name
    for class_name_index in range(1, len(parts) + 1):
        try:
            if class_name_index < len(parts):
                attr = __import__(
                    '.'.join(parts[:class_name_index]),
                    fromlist=[parts[class_name_index]],
                )
                for part in parts[class_name_index:]:
                    attr = getattr(attr, part)
            else:
                attr = __import__('.'.join(parts))

            LOG.debug(f'{name} imported. Found {attr}')
            return attr

        except (ModuleNotFoundError, AttributeError):
            LOG.debug(
                f'{name}: import failed: {".".join(parts[class_name_index:])}'
                f' from package {".".join(parts[:class_name_index])}'
            )

    raise ComponentNotFoundError(name)


class ComponentNotFoundError(ValueError):
    pass


class InvalidRegistryEntryError(ValueError):
    pass


class ConfigParserLoader(ABC, Iterable):
    """
    Base class for a configuration parser loader.

    A configuration parser is able to interpret a dictionary-style configuration loaded from a YAML. Sub classes are
    expected to implement the `get()` method.
    """

    @staticmethod
    def _get_config_parser(
        result_type: Any, default_config_parser: Callable[[Any], ConfigParser] | None
    ) -> ConfigParser:
        if inspect.isclass(result_type) and issubclass(result_type, ConfigParser):
            return result_type()
        elif default_config_parser is not None:
            return default_config_parser(result_type)
        else:
            raise InvalidRegistryEntryError(
                f'unable to instantiate {result_type}: '
                'not a ConfigParser and there is no default configuration parser in this context'
            )

    @abstractmethod
    def get(
        self,
        key: str,
        default_config_parser: Callable[[Any], ConfigParser] | None = None,
        search_path: list[str] | None = None,
    ) -> ConfigParser:
        """
        Retrieve a value for a given key name.

        The key may resolve to a `ConfigParser` class, or it is passed as an argument to `default_config_parser`, which
        in turn returns a `ConfigParser` class.

        :param key: the key name to resolve
        :param default_config_parser: a function that returns a `ConfigParser` if the `key` does not resolve to a
            `ConfigParser`
        :param search_path: the domain of the search query
        :return: a `ConfigParser` object
        """
        raise NotImplementedError


class ClassLoader(ConfigParserLoader):
    """
    A configuration parser loader that uses reflection to resolve class names.
    """

    def __iter__(self) -> Iterator[str]:
        return iter([])

    def get(
        self,
        key: str,
        default_config_parser: Callable[[Any], ConfigParser] | None = None,
        search_path: list[str] | None = None,
    ) -> ConfigParser:
        parts = key.split('.')
        if len(parts) < 2:
            raise ComponentNotFoundError(f'no full class name: {key}')

        try:
            result_type = _get_attribute_by_name(key)
        except AttributeError as e:
            raise ComponentNotFoundError(str(e))
        except ModuleNotFoundError as e:
            raise ComponentNotFoundError(str(e))

        return ConfigParserLoader._get_config_parser(result_type, default_config_parser)


class FederatedLoader(ConfigParserLoader):
    """
    A configuration parser loader that delegates resolution to other loaders.
    """

    def __init__(self, registries: list[ConfigParserLoader]):
        self.registries = registries

    def __iter__(self) -> Iterator[str]:
        for r in self.registries:
            yield from r

    def get(
        self,
        key: str,
        default_config_parser: Callable[[Any], ConfigParser] | None = None,
        search_path: list[str] | None = None,
    ) -> ConfigParser:
        errors = []
        for r in self.registries:
            try:
                LOG.debug(f'trying to load {key} from {r} (search_path={search_path})')
                return r.get(key, default_config_parser, search_path)
            except ComponentNotFoundError as e:
                errors.append(e)

        raise ComponentNotFoundError('; '.join([str(e) for e in errors]))


def _load_package_registry() -> 'YamlRegistry':
    registry_file = importlib.resources.files(package_resources) / 'registry.yaml'
    with registry_file.open('r') as f:
        LOG.debug(f'loading registry from package resource: {registry_file}')
        lib_registry = confidence.load(f)

    user_registry = confidence.load_name('registry')
    merged_registry = confidence.Configuration(lib_registry, user_registry)

    LOG.debug(f'loaded registry with entries: {list(YamlRegistry(merged_registry))}')

    return YamlRegistry(merged_registry)


def registry() -> ConfigParserLoader:
    """Provide access to a centralized registry of available configuration options."""
    global _REGISTRY
    if _REGISTRY is None:
        try:
            yaml_registry = _load_package_registry()
            _REGISTRY = FederatedLoader([yaml_registry, ClassLoader()])
        except Exception as e:
            raise ValueError(f'registry initialization failed: {e}')

    return _REGISTRY


def get(
    name: str,
    default_config_parser: Callable[[Any], ConfigParser] | None = None,
    search_path: list[str] | None = None,
) -> ConfigParser:
    """Retrieve corresponding value for a given key name from the central registry."""
    return registry().get(name, default_config_parser, search_path)


class YamlRegistry(ConfigParserLoader):
    """Representation of a YAML-based registry.

    The YAML registry is expected to define "sections" as the top-level
    key names, followed by keys referring to (paths to) classnames or
    functions.

    This registry parses this YAML mapping and provides access to these
    values through a `get()` method.
    """

    def __init__(self, cfg: confidence.Configuration):
        self._cfg = cfg

    def __iter__(self) -> Iterator[str]:
        for toplevel in self._cfg:
            for component in self._cfg.get(toplevel):
                yield f'{toplevel}.{component}'

    def __str__(self) -> str:
        return f'YamlRegistry({self._cfg})'

    @staticmethod
    def _parse(
        key: str, spec: Mapping[str, str], default_config_parser: Callable[[Any], ConfigParser] | None
    ) -> ConfigParser:
        if 'class' not in spec:
            raise InvalidRegistryEntryError(f'missing value for `class` in registry entry: {key}')
        if not isinstance(spec.get('class'), str):
            raise InvalidRegistryEntryError(
                f'expected `str` type for `class` in registry entry: {key}; found: {type(spec.get("class"))}'
            )

        try:
            cls = _get_attribute_by_name(spec.get('class'))  # type: ignore[arg-type]
        except Exception as e:
            raise ValueError(f'registry key `{key}` resolved to `{spec.get("class")}` but failed to materialize: {e}')

        parser = ConfigParserLoader._get_config_parser(cls, default_config_parser)

        if 'wrapper' in spec:
            try:
                wrapper = _get_attribute_by_name(spec.get('wrapper'))  # type: ignore[arg-type]
            except Exception as e:
                raise InvalidRegistryEntryError(
                    f'unable to instantiate class {spec["class"]}: '
                    f'error while instantiating wrapper class: {spec["wrapper"]}: {e}'
                )
            parser = wrapper(parser)

        return parser

    def _find(self, key: str, search_path: list[str] | None) -> Any:
        """Locate the value for a given key name in the YAML-based registry.

        The search path is used to prefix the key name with possible
        domain (for example: 'modules' or 'data_provider')."""
        try_keys = [key]

        if search_path is not None:
            try_keys += [f'{path_prefix}.{key}' for path_prefix in search_path]

        for try_key in try_keys:
            if try_key in self._cfg:
                LOG.debug(f'{try_key}: registry entry found')
                return self._cfg.get(try_key)

        raise ComponentNotFoundError(f'component not found: {key} (tried: {", ".join(try_keys)})')

    def get(
        self,
        key: str,
        default_config_parser: Callable[[Any], ConfigParser] | None = None,
        search_path: list[str] | None = None,
    ) -> ConfigParser:
        """Retrieve a value for a given key name from the YAML-based registry.

        An entry can take the following forms, available under the keys `path.to.key1` and
        `path.to.key2` respectively:
        ```
        path.to.key1: ObjectName
        path.to.key2:
            class: ObjectName
        ```

        In the example, `ObjectName` refers to a Python object available in the current runtime.
        """
        spec = self._find(key, search_path)
        if isinstance(spec, str):
            return self._parse(key, {'class': spec}, default_config_parser)
        else:
            return self._parse(key, spec, default_config_parser)


_REGISTRY: ConfigParserLoader | None = None
