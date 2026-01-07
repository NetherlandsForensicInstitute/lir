from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from lir import registry
from lir.transform.pairing import PairingMethod


class YamlParseError(ValueError):
    """Error raised when parsing YAML configuration fails, mentioning specific YAML path."""

    def __init__(self, config_context_path: list[str], message: str):
        prefix = f'{".".join(config_context_path)}: ' if config_context_path else ''
        super().__init__(f'{prefix}{message}')


class ContextAwareDict(dict):
    def __init__(self, context: list[str], *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.context = context

    def clone(self, context: list[str] | None = None) -> 'ContextAwareDict':
        return _expand(context if context is not None else self.context, self)


class ContextAwareList(list):
    def __init__(self, context: list[str], *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.context = context

    def clone(self, context: list[str] | None = None) -> 'ContextAwareList':
        return _expand(context if context is not None else self.context, self)


def _expand(context: list[str], cfg: Any) -> Any:
    """Iteratively unpack the data structure into the appropriate underlying representation."""
    if isinstance(cfg, Mapping):
        return ContextAwareDict(context, [(key, _expand(context + [key], value)) for key, value in cfg.items()])
    elif isinstance(cfg, str):
        return cfg
    elif isinstance(cfg, Sequence):
        return ContextAwareList(context, [_expand(context + [str(i)], value) for i, value in enumerate(cfg)])
    return cfg


class ConfigParser(ABC):
    """Abstract base configuration parser class.

    Each implementation should implement a custom `parse()` method
    which is dedicated to parsing a specific aspect, e.g. the configuration
    for setting up the numpy CSV writer.
    """

    @abstractmethod
    def parse(
        self,
        config: ContextAwareDict,
        output_dir: Path,
    ) -> Any:
        """Dedicated function to parse a specific section of a YAML configuration.

        Arguments:
        - config: a section of a YAML configuration
        - config_context_path: the path in the YAML configuration to `config`, the section to be parsed
        - output_dir: the directory where the returned object may write results during its lifetime

        Returns: an object that is configured according to `config`.
        """
        raise NotImplementedError


class GenericFunctionConfigParser(ConfigParser):
    """Parser for callable functions or component classes."""

    def __init__(self, component_class: Any):
        super().__init__()
        self.component_class = component_class

    def parse(
        self,
        config: ContextAwareDict,
        output_dir: Path,
    ) -> Callable:
        if callable(self.component_class):
            return self.component_class

        raise YamlParseError(config.context, f'unrecognized module type: `{self.component_class}`')


class GenericConfigParser(ConfigParser):
    """
    This parser returns an instantiation of a class, initialized with the specified arguments.
    """

    def __init__(self, component_class: Any):
        super().__init__()
        self.component_class = component_class

    def parse(
        self,
        config: ContextAwareDict,
        output_dir: Path,
    ) -> Any:
        try:
            return self.component_class(**config)
        except Exception as e:
            raise YamlParseError(
                config.context,
                f'unable to initialize {self.component_class}; the error was: {e}',
            )


def config_parser(func: Callable) -> Callable:
    """Decorator function to wrap parsing functions in a `ConfigParser` object.

    The `ConfigParser` object exposes a `parse()` method, required by the API.

    Using the `@config_parser` decorator, exposes the body of the function through the wrapped
    `parse()` method.

    This decorator can be used as follows (example):
    ```
    @config_parser
    def foo(config, config_context_path, output_dir):
      if "some_argument" not in config or "another_argument" not in config:
        raise YamlParseError(config_context_path, "a required argument is missing")
      return Bar(config["some_argument"], config["another_argument"])
    ```

    Now, the function `foo()` is wrapped within a `ConfigParser` object,
    which exposes the function body of `foo()` through the `parse()` method.
    See documentation of `ConfigParser` for the meaning of the arguments.
    """

    class ConfigParserFunction(ConfigParser):
        def parse(
            self,
            config: ContextAwareDict,
            output_dir: Path,
        ) -> Any:
            return func(config, output_dir)

    return ConfigParserFunction


def parse_pairing_config(
    module_config: ContextAwareDict | str,
    output_dir: Path,
    context: list[str],
) -> PairingMethod:
    """Generic parser function for the pairing setup.

    This function will not return a pairing module itself. Instead, it delegates this
    task to the config parser of the pairing method of choice.
    The argument `module_config` defines the pairing method. If its value is a `str`, the registry is queried and the
    corresponding pairing method is returned. If its value is a `dict`, the pairing method is defined
    by the value `module_config["method"]`, and the registry is queried for the config parser of
    the corresponding pairing method. The remaining values in `module_config` are passed as arguments to the
    configuration parser of the pairing method.

    If the registry cannot resolve the pairing method, an exception is raised.
    """
    if isinstance(module_config, str):
        class_name = module_config
        args = ContextAwareDict(context)
    else:
        class_name = pop_field(module_config, 'method')
        args = module_config

    return registry.get(class_name, search_path=['pairing'], default_config_parser=GenericConfigParser).parse(
        args, output_dir
    )


def check_not_none(v: Any) -> Any:
    if v is None:
        raise ValueError('value None is not allowed here')
    return v


YamlValueType = ContextAwareDict | ContextAwareList | None | int | float | str


_YAML_TYPES: dict = {
    ContextAwareDict: dict,
    ContextAwareList: list,
}


def check_type(type_class: Any, v: YamlValueType, message: str | None = None) -> Any:
    """
    Check whether a value is an instance of a type.

    Returns the value if successful, raises an exception otherwise.

    Value types that may be found in YAML configurations:
    - dict
    - list
    - int
    - float
    - str
    - NoneType

    :param type_class: the target type
    :param v: the value to check
    :param message: an optional message that is used in case of an error
    :return: the value
    """
    if isinstance(v, type_class):
        return v
    else:
        message = message or f'expected type: {type_class.__name__}'
        actual_type = _YAML_TYPES.get(type(v), type(v))  # translate types to python built-in types
        raise ValueError(f'{message}; found: {actual_type.__name__}')


def pop_field(
    config: ContextAwareDict | Any,
    field: str,
    default: Any = None,
    required: bool | None = None,
    validate: Callable[[Any], Any] | None = None,
) -> Any:
    """
    Validate and retrieve the value for a given field, after which it is removed from the configuration.

    :param config: the configuration
    :param field: the field to obtain from the `config`
    :param default: the value to return if the field is not found; defaults to `None`; if the value is not `None`, the
        `required` argument defaults to `False`
    :param required: if `True` and the field was not found, raise an error; defaults to `True` unless `default` is not
        `None`
    :param validate: a callable to validate the value type
    :return: the field value or the default value or an error is raised
    """
    # get required status and default value from function arguments
    required = required if required is not None else (default is None)
    if default is not None and required:
        raise ValueError(f'illegal argument values: required={required}; default={default}')

    # if there is a configuration, it should be a `dict`, and we will try to get the field value from it
    if config:
        if not isinstance(config, ContextAwareDict):
            raise YamlParseError(config.context, f'expected dict; found: {type(config)}')
        elif field in config:
            value = config.pop(field)
            if validate:
                try:
                    value = validate(value)
                except Exception as e:
                    raise YamlParseError(config.context, f'illegal value for field `{field}`: {e}')
            return value

    # if no field value was returned, return the default value or raise an error
    if required:
        raise YamlParseError(config.context, f'missing field: `{field}`')
    else:
        return default


def check_is_empty(
    config: ContextAwareDict,
    accept_keys: Sequence[str] | None = None,
) -> None:
    """Ensure all defined expected arguments are parsed and warn about ignored arguments.

    If any unexpected arguments remain, a `YamlParseError` is raised indicating the
    argument was unexpected and not taken into account (i.e. not parsed). This methodology ensures
    the user does not assume arguments are parsed that are in fact not recognized."""
    for key in config:
        if not accept_keys or key not in accept_keys:
            raise YamlParseError(config.context, f'unrecognized argument: {key}')
