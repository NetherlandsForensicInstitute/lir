from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence

from typing import Any, Mapping, Callable

from lir import registry
from lir.transform.pairing import PairingMethod


class YamlParseError(ValueError):
    """Error raised when parsing YAML configuration fails, mentioning specific YAML path."""

    def __init__(self, config_context_path: list[str], message: str):
        prefix = f"{'.'.join(config_context_path)}: " if config_context_path else ""
        super().__init__(f"{prefix}{message}")


class ConfigParser(ABC):
    """Abstract base configuration parser class.

    Each implementation should implement a custom `parse()` method
    which is dedicated to parsing a specific aspect, e.g. the configuration
    for setting up the numpy CSV writer.
    """

    @abstractmethod
    def parse(
        self,
        config: dict[str, Any],
        config_context_path: list[str],
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
        config: Mapping[str, Any],
        config_context_path: list[str],
        output_dir: Path,
    ) -> Callable:
        if callable(self.component_class):
            return self.component_class

        raise YamlParseError(
            config_context_path, f"unrecognized module type: `{self.component_class}`"
        )


class GenericConfigParser(ConfigParser):
    """
    This parser returns an instantiation of a class, initialized with the specified arguments.
    """

    def __init__(self, component_class: Any):
        super().__init__()
        self.component_class = component_class

    def parse(
        self,
        config: Mapping[str, Any],
        config_context_path: list[str],
        output_dir: Path,
    ) -> Any:
        try:
            return self.component_class(**config)
        except Exception as e:
            raise YamlParseError(
                config_context_path,
                f"unable to initialize {self.component_class}; the error was: {e}",
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
            config: Mapping[str, Any],
            config_context_path: list[str],
            output_dir: Path,
        ) -> Any:
            return func(config, config_context_path, output_dir)

    return ConfigParserFunction


def parse_pairing_config(
    module_config: dict[str, Any] | str,
    config_context_path: list[str],
    output_dir: Path,
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
        args = {}
    else:
        class_name = pop_field(config_context_path, module_config, "method")
        args = module_config

    return registry.get(
        class_name, search_path=["pairing"], default_config_parser=GenericConfigParser
    ).parse(args, config_context_path, output_dir)


def get_parser_arguments_for_field(
    config: dict[str, Any], context: list[str], output_path: Path, field: str
) -> Any:
    """Initialize the appropriate parser for a given field in the YAML configuration (e.g. 'module')."""
    value = pop_field(context, config, field)
    return value, context + [field], output_path


def pop_field(
    context_path: list[str],
    config: dict[str, Any] | Any,
    field: str,
    default: Any = None,
    required: bool | None = None,
    validate: Callable | None = None,
) -> Any:
    """
    Validate and retrieve the value for a given field, after which it is removed from the configuration.

    :param context_path: the configuration context path
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
        raise ValueError(
            f"illegal argument values: required={required}; default={default}"
        )

    # if there is a configuration, it should be a `dict`, and we will try to get the field value from it
    if config:
        if not isinstance(config, dict):
            raise YamlParseError(context_path, f"expected dict; found: {type(config)}")
        elif field in config.keys():
            value = config.pop(field)
            if validate:
                try:
                    value = validate(value)
                except Exception as e:
                    raise YamlParseError(
                        context_path, f"illegal value for field `{field}`: {e}"
                    )
            return value

    # if no field value was returned, return the default value or raise an error
    if required:
        raise YamlParseError(context_path, f"missing field: `{field}`")
    else:
        return default


def check_is_empty(
    config_context_path: list[str],
    config: dict[str, Any],
    accept_keys: Optional[Sequence[str]] = None,
) -> None:
    """Ensure all defined expected arguments are parsed and warn about ignored arguments.

    If any unexpected arguments remain, a `YamlParseError` is raised indicating the
    argument was unexpected and not taken into account (i.e. not parsed). This methodology ensures
    the user does not assume arguments are parsed that are in fact not recognized."""
    for key in config:
        if not accept_keys or key not in accept_keys:
            raise YamlParseError(config_context_path, f"unrecognized argument: {key}")
