import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

from lir.util import check_type


class YamlParseError(ValueError):
    """
    Error raised when parsing YAML configuration fails, mentioning specific YAML path.

    Parameters
    ----------
    config_context_path : list[str]
        Dot-path to the failing configuration node.
    message : str
        Human-readable validation or parsing message.
    """

    def __init__(self, config_context_path: list[str], message: str):
        prefix = f'{".".join(config_context_path)}: ' if config_context_path else ''
        super().__init__(f'{prefix}{message}')


@dataclass
class ConfigValue:
    """
    A wrapper for a configuration value and its context path.

    This configuration value may be part of a bigger configuration tree. For example, consider:

    .. code-block:: yaml

        root:
          value1:
            a: 1
            b: 2
          value2:
            c: 3

    This YAML is parsed into a dictionary and the path ``root.value1`` leads to the value ``{ "a": 1, "b": 2 }``. In a
    ``ConfigValue`` object, this is represented as context path ``["root", "value1"]`` and value ``{ "a": 1, "b": 2 }``.

    Attributes
    ----------
    context : list[str]
        YAML path used for contextual error messages.
    value : list[ConfigValue] | dict[str, ConfigValue] | int | float | bool | str | None
        The actual configuration value.
    """

    context: list[str]
    value: 'list[ConfigValue] | dict[str, ConfigValue] | int | float | bool | str | None'

    def check_type[ValueType: Any](
        self,
        type_class: type[ValueType],
        message: str | None = None,
        unwrap: bool = True,
    ) -> ValueType:
        """
        Check value type.

        Parameters
        ----------
        type_class : type
            The type to assert for the value of this object.
        message : str | None
            Human-readable validation or parsing error message in case of failure.
        unwrap : bool
            If ``True``, unwrap the value of this object recursively.

        Returns
        -------
        ValueType
            The value of this object. Any nested values are still wrapped in ``ConfigValue`` objects.
        """
        try:
            check_type(type_class, self.value, message)
            return self.unwrap() if unwrap else self.value  # type: ignore
        except ValueError as e:
            raise YamlParseError(self.context, str(e))

    def __iter__(self) -> 'Iterator[str | ConfigValue]':
        return iter(check_type((dict, list), self.value))

    def __contains__(self, item: str | int) -> bool:
        return isinstance(self.value, (dict, list)) and item in self.value

    def __getitem__(self, item: str | int) -> 'ConfigValue':
        if isinstance(self.value, dict) and isinstance(item, str):  # noqa: SIM114
            return self.value[item]
        elif isinstance(self.value, list) and isinstance(item, int):
            return self.value[item]
        else:
            raise ValueError(f'type not indexable by {item}: {type(item)}')

    def __setitem__(self, item: str | int, value: Any) -> None:
        if isinstance(value, ConfigValue):
            value = value.unwrap()
        if isinstance(self.value, dict) and isinstance(item, str):
            self.value[item] = ConfigValue.wrap(self.context + [item], value)
        elif isinstance(self.value, list) and isinstance(item, int):
            self.value[item] = ConfigValue.wrap(self.context + [str(item)], value)
        else:
            raise ValueError(f'type not indexable by {item}: {type(item)}')

    def unwrap(self) -> list | dict | int | float | bool | str | None:
        """
        Obtain the value of this object.

        If the value is a container, its contents are also stripped of its ``ConfigValue`` wrapper recursively.

        Returns
        -------
        list | dict | int | float | bool | str | None
            The value of this object.
        """
        if isinstance(self.value, list):
            return [value.unwrap() for value in self.value]
        elif isinstance(self.value, dict):
            return {k: v.unwrap() for k, v in self.value.items()}
        else:
            return self.value

    def as_dict(self) -> dict:  # numpydoc ignore=RT01
        """Return unwrapped dictionary or raise an error."""
        return check_type(dict, self.unwrap())

    def clone(self, context: list[str] | None = None) -> 'ConfigValue':
        """
        Create a cloned list with expanded nested context.

        Parameters
        ----------
        context : list[str] | None, optional
            Replacement context. If omitted, the current context is reused.

        Returns
        -------
        ConfigValue
            Cloned and context-aware list.
        """
        return ConfigValue.wrap(context or self.context, self.unwrap())  # type: ignore

    @staticmethod
    def wrap(context: list[str], value: Sequence | Mapping | int | float | bool | str | None) -> 'ConfigValue':
        """
        Wrap a value and all its nested values into :class:`~lir.config.base.ConfigValue` objects, recursively.

        Parameters
        ----------
        context : list[str]
            Current YAML path.
        value : Sequence | Mapping | float | int | str | None
            Value to expand recursively.

        Returns
        -------
        ConfigValue
            The value wrapped into ``ConfigValue`` objects recursively.
        """
        if isinstance(value, Mapping):
            return ConfigValue(context, {key: ConfigValue.wrap(context + [key], value) for key, value in value.items()})
        elif isinstance(value, str):
            return ConfigValue(context, value)
        elif isinstance(value, Sequence):
            return ConfigValue(context, [ConfigValue.wrap(context + [str(i)], value) for i, value in enumerate(value)])
        return ConfigValue(context, value)


class ConfigParser(ABC):
    """
    Abstract base configuration parser class.

    Each implementation should implement a custom `parse()` method
    which is dedicated to parsing a specific aspect, e.g. the configuration
    for setting up the numpy CSV writer.
    """

    @abstractmethod
    def parse(
        self,
        config: ConfigValue,
        output_dir: Path,
    ) -> Any:
        """
        Parse a specific configuration section.

        Parameters
        ----------
        config : ConfigValue
            Configuration section to parse.
        output_dir : Path
            Directory where produced outputs may be written.

        Returns
        -------
        Any
            Object configured from ``config``.
        """
        raise NotImplementedError

    @staticmethod
    def get_type_name(obj: Any) -> str:
        """
        Return the fully qualified type name.

        Parameters
        ----------
        obj : Any
            Class or object with ``__module__`` and ``__qualname__`` attributes.

        Returns
        -------
        str
            Fully qualified name.
        """
        module = obj.__module__
        return f'{module}.{obj.__qualname__}'

    def reference(self) -> str:
        """
        Return the full class name that was used to initialize this parser.

        By default, return the name of this class. In a subclass that was initialized with another class or function
        that does the actual work, the name of that class is returned.

        Returns
        -------
        str
            Fully qualified class name for this parser instance.
        """
        return self.get_type_name(self.__class__)


class GenericFunctionConfigParser(ConfigParser):
    """
    Parser for callable functions or component classes.

    Parameters
    ----------
    component_class : Callable
        Callable that should be exposed by this parser.
    """

    def __init__(self, component_class: Callable):
        super().__init__()
        self.component_class = component_class

    def parse(
        self,
        config: ConfigValue,
        output_dir: Path,
    ) -> Callable:
        """
        Parse configuration into a callable.

        Parameters
        ----------
        config : ConfigValue
            Configuration section for validation context.
        output_dir : Path
            Unused output directory argument required by the parser API.

        Returns
        -------
        Callable
            Resolved callable object.
        """
        if callable(self.component_class):
            return self.component_class

        raise YamlParseError(config.context, f'unrecognized module type: `{self.component_class}`')

    def reference(self) -> str:
        """
        Return the fully qualified name of the wrapped callable.

        Returns
        -------
        str
            Fully qualified callable name.
        """
        return self.get_type_name(self.component_class)


class GenericConfigParser(ConfigParser):
    """
    Return an instantiation of a class, initialized with the specified arguments.

    Parameters
    ----------
    component_class : type[Any]
        Class to instantiate from configuration values.
    """

    def __init__(self, component_class: type[Any]):
        super().__init__()
        self.component_class = component_class

    def parse(
        self,
        config: ConfigValue,
        output_dir: Path,
    ) -> Any:
        """
        Instantiate the configured component class.

        Parameters
        ----------
        config : ConfigValue
            Keyword arguments for class initialisation.
        output_dir : Path
            Unused output directory argument required by the parser API.

        Returns
        -------
        Any
            Instantiated object.
        """
        try:
            return self.component_class(**config.as_dict())
        except Exception as e:
            raise YamlParseError(
                config.context,
                f'unable to initialize {self.component_class}; the error was: {e}',
            )

    def reference(self) -> str:
        """
        Return the fully qualified name of the wrapped class.

        Returns
        -------
        str
            Fully qualified class name.
        """
        return self.get_type_name(self.component_class)


def get_full_name(obj: Any) -> str:
    """
    Return the full name of an importable object.

    .. code-block:: python

        from lir import FeatureData
        print(get_full_name(FeatureData))
        'lir.FeatureData'

    Parameters
    ----------
    obj : Any
        Importable object.

    Returns
    -------
    str
        Fully qualified object name.
    """
    return f'{obj.__module__}.{obj.__name__}'


def config_parser(
    func: Callable[[ConfigValue, Path], Any] | None = None, /, reference: str | Any | None = None
) -> Callable:
    """
    Wrap a parsing function in a ``ConfigParser`` object using a decorator.

    The resulting ``ConfigParser`` instance exposes a :meth:`parse` method, as
    required by the API. The body of the decorated function is executed when the
    :meth:`parse` method is called.

    This decorator can be used as follows:

    .. code-block:: python

        @config_parser
        def foo(config, config_context_path, output_dir):
            if "some_argument" not in config or "another_argument" not in config:
                raise YamlParseError(
                    config_context_path,
                    "a required argument is missing",
                )
            return Bar(config["some_argument"], config["another_argument"])

    After decoration, ``foo`` is replaced by a ``ConfigParser`` instance whose
    :meth:`parse` method executes the original function body. See the
    documentation of :class:`ConfigParser` for the meaning of the arguments.

    The annotated function will be the reference object that users will be referred to for documentation. If the
    annotation has a `reference` argument, that value will be used instead. The `reference` value may be a `str` or a
    Python object. Example of use:

    .. code-block:: python

        @config_parser(reference=Bar)
        def foo(config, config_context_path, output_dir):
            if "some_argument" not in config or "another_argument" not in config:
                raise YamlParseError(
                    config_context_path,
                    "a required argument is missing",
                )
            return Bar(config["some_argument"], config["another_argument"])

    Parameters
    ----------
    func : Callable[[ConfigValue, Path], Any] | None, optional
        Function to wrap as a config parser.
    reference : str | Any | None, optional
        Explicit reference name or object used in generated metadata.

    Returns
    -------
    Callable
        Decorator result or wrapped ``ConfigParser`` implementation.
    """
    if func is None:
        # take the optional arguments
        return partial(config_parser, reference=reference)

    class ConfigParserFunction(ConfigParser):
        __doc__ = func.__doc__

        def parse(
            self,
            config: ConfigValue,
            output_dir: Path,
        ) -> Any:
            return func(config, output_dir)  # type: ignore

        def reference(self) -> str:
            # return the reference argument, if any
            if reference is not None:
                return reference if isinstance(reference, str) else get_full_name(reference)

            if func is None:  # at this point, func is always available
                raise RuntimeError('unexpected error: function is not available for reference')

            # return the return type of the function, if available
            return_type = inspect.signature(func).return_annotation
            if not isinstance(return_type, str):
                return get_full_name(return_type)

            # last resort: fallback to wrapped function name
            return get_full_name(func)

    return ConfigParserFunction


AnyType = TypeVar('AnyType')


def check_not_none[AnyType](v: AnyType | None, message: str | None = None) -> AnyType:
    """
    Validate a value is not ``None``.

    Parameters
    ----------
    v : AnyType | None
        Value to validate.
    message : str | None, optional
        Error message used when ``v`` is ``None``.

    Returns
    -------
    AnyType
        Original non-``None`` value.
    """
    if v is None:
        raise ValueError(message or 'value None is not allowed here')
    return v


def pop_field(
    config: ConfigValue,
    field: str,
    default: Any = None,
    required: bool | None = None,
    validate: Callable[[Any], Any] | None = None,
    validate_type: type[Any] | None = None,
    unwrap: bool | None = None,
) -> Any:
    """
    Validate and retrieve the value for a given field, after which it is removed from the configuration.

    Parameters
    ----------
    config : ConfigValue
        Configuration object to pop from.
    field : str
        Field name to retrieve.
    default : Any, optional
        Value to return when ``field`` is absent.
    required : bool | None, optional
        Whether to raise when the field is absent. Defaults to ``True`` when
        ``default`` is ``None``.
    validate : Callable[[Any], Any] | None, optional
        Validator function applied to the popped value.
    validate_type : type[Any] | None, optional
        Check that the popped value is of this type, or raise a ``ValueError``.
    unwrap : bool | None, optional
        Strip the popped value of its :class:`~lir.config.base.ConfigValue` wrapper before returning it. Defaults to
        ``True`` if either ``validate`` or ``validate_type`` or ``default`` is provided, except if the default is a
        ``Config|Value``. Defaults to ``False`` otherwise.

    Returns
    -------
    Any
        Popped field value or ``default``.
    """
    if validate is not None and validate_type is not None:
        raise ValueError('illegal combination of `validate` and `validate_type`')

    # get required status and default value from function arguments
    required = required if required is not None else (default is None)
    if default is not None and required:
        raise ValueError(f'illegal argument values: required={required}; default={default}')

    if unwrap is None:
        if default is not None and isinstance(default, ConfigValue):
            unwrap = False
        elif validate is not None or validate_type is not None or default is not None:
            unwrap = True
        else:
            unwrap = False

    # if there is a configuration, it should be a `dict`, and we will try to get the field value from it
    if field in config.check_type(dict):
        value = config.value.pop(field)  # type: ignore

        try:
            if validate_type is not None:
                check_type(validate_type, value.value)  # type: ignore
            if unwrap:
                value = value.unwrap()
            if validate:
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
    config: ConfigValue,
    accept_keys: Sequence[str] | None = None,
) -> None:
    """
    Ensure all defined expected arguments are parsed and warn about ignored arguments.

    If any unexpected arguments remain, a `YamlParseError` is raised indicating the
    argument was unexpected and not taken into account (i.e. not parsed). This methodology ensures
    the user does not assume arguments are parsed that are in fact not recognized.

    Parameters
    ----------
    config : ConfigValue
        Configuration to validate for remaining keys.
    accept_keys : Sequence[str] | None, optional
        Keys that may remain without raising an error.

    Returns
    -------
    None
        This function raises on invalid input and otherwise returns ``None``.
    """
    if config.value is None or (isinstance(config.value, list) and len(config.value) == 0):
        return
    if isinstance(config.value, dict):
        for key in config.value:
            if not accept_keys or key not in accept_keys:
                raise YamlParseError(config.context, f'unrecognized argument: {key}')
