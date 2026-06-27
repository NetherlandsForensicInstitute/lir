import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from lir.util import check_type
from types import UnionType
from typing import Any, NamedTuple, TypeVar


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

    A ``ConfigValue`` has two attributes: ``context`` contains the path in the original configuration that points to the
    current value, and ``value`` contains the actual configuration value. The value attribute may be one of the types
    allowed in YAML: ``dict``, ``list``, ``int``, ``float``, ``bool``, ``str``, or ``None``.

    This configuration value may be part of a bigger configuration tree. For example, consider:

    .. code-block:: yaml

        level1:
          value1:
            a: 1
            b: 2
          value2:
            c: 3
            d: path/to/file
            e: message text

    This YAML is parsed into a dictionary and the path ``level1.value1`` leads to the value ``{ "a": 1, "b": 2 }``. In a
    ``ConfigValue`` object, this is represented as context path ``["level1", "value1"]`` and value
    ``{ "a": 1, "b": 2 }``.

    If the ``value`` is dictionary, its values are itself also ``ConfigValue`` objects. They can be obtained using the
    helper function :meth:`~lir.config.pop_field`. When all is done, :meth:`~lir.config.check_empty` will check that
    all values have been read.

    Some examples for the use of ``ConfigValue``:

    .. jupyter-execute::

        from lir.config import ConfigValue, pop_field, check_is_empty

        my_config_input = {
            'level1': {
                'value1': {
                    'a': 1,
                    'b': 2,
                },
                'value2': {
                    'c': 3,
                    'd': 'path/to/file',
                    'e': 'message text',
                },
            },
        }

        root_config = ConfigValue.wrap([], my_config_input)
        print(f'The context of root is: {root_config.context}')
        print(f'The value of root is: {root_config.value}')
        print(f'The unwrapped value of root is: {root_config.unwrap()}')

    Example for the use of :meth:`~lir.config.check_empty`:

    .. jupyter-execute::

        level1_config = pop_field(root_config, 'level1')
        print(f'After "level1" is popped, the value of root is: {root_config.value}')

        # We can call `check_is_empty()` to make sure that there are no values left.
        check_is_empty(root_config)

        print(f'The context of level1 is: {level1_config.context}')
        print(f'The value of level1 is: {level1_config.value}')
        print(f'The unwrapped value of level1 is: {level1_config.unwrap()}')

    Example for the use of ``with``:

    .. jupyter-execute::

        # use the `with` statement to automatically check that all fields are used
        try:
            with pop_field(level1_config, 'value1') as value1_config:
                pop_field(value1_config, 'a')
        except Exception as e:
            print(f'Exception thrown: {e}')

    More examples:

    .. jupyter-execute::

        value2_config = pop_field(level1_config, 'value2')
        print(f'The context of level1.value2 is: {value2_config.context}')
        print(f'The value of level1.value2 is: {value2_config.value}')
        print(f'The unwrapped value of level1.value2 is: {value2_config.unwrap()}')

    Examples for the use of :meth:`~lir.config.pop_field`:

    .. jupyter-execute::

        from pathlib import Path

        # use `validate_type` to check that the type is as expected
        c = pop_field(value2_config, 'c', validate_type=int)
        print(f'The value of level1.value2.c is: {c}')

        # use `validate` to cast types or call a function to otherwise process the value
        d = pop_field(value2_config, 'd', validate=Path)
        print(f'The value of level1.value2.d is: {d}')

        # use `unwrap` to control whether the result should be unwrapped or not
        e = pop_field(value2_config, 'e', unwrap=False)
        print(f'The value of level1.value2.e is: {e}')
    """

    context: list[str]
    """YAML path used for contextual error messages."""

    value: 'list[ConfigValue] | dict[str, ConfigValue] | int | float | bool | str | None'
    """The actual configuration value."""

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
            raise ValueError(f'type {type(self)} is not indexable by {item} ({type(item)})')

    def __setitem__(self, item: str | int, value: Any) -> None:
        if isinstance(value, ConfigValue):
            value = value.unwrap()
        if isinstance(self.value, dict) and isinstance(item, str):
            self.value[item] = ConfigValue.wrap(self.context + [item], value)
        elif isinstance(self.value, list) and isinstance(item, int):
            self.value[item] = ConfigValue.wrap(self.context + [str(item)], value)
        else:
            raise ValueError(f'type {type(self)} is not indexable by {item} ({type(item)})')

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if not exception_value:
            check_is_empty(self)
        return None

    def pop(
        self,
        field: str,
        default: Any = None,
        required: bool | None = None,
        validate: Callable[[Any], Any] | None = None,
        validate_type: type[Any] | None = None,
    ) -> 'ConfigValue | None':
        """
        Validate and retrieve the value for a given field, after which it is removed from this configuration.

        If the value of this ``ConfigValue`` is not a ``dict``, an error is raised.

        If the field exists, it is returned as a ``ConfigValue`` object.

        If the field does not exist, and a default is provided, the default is returned, wrapped in a ``ConfigValue``
        object.

        If the field does not exist, it is optional, and no default is provided, ``None`` is returned.

        Otherwise, the field does not exist, and it is required: an error is raised.

        This method behaves similarly to ``pop_field()``, except that its return value is wrapped in ``ConfigValue``.

        Parameters
        ----------
        field : str
            Field name to retrieve.
        default : Any, optional
            Value to return when ``field`` is absent.
        required : bool | None, optional
            Whether to raise when the field is absent. Defaults to ``True`` when
            ``default`` is ``None``.
        validate : Callable[[Any], Any] | None, optional
            Validator function applied to the popped value. The output from the validation function is returned, in
            place of the original value.
        validate_type : type[Any] | None, optional
            Check that the popped value is of this type, or raise a ``ValueError``.

        Returns
        -------
        ConfigValue
            Popped field value or ``default``.
        """
        # this value should be a dict
        dict_value = check_type(dict, self.value)

        # get required status and default value from function arguments
        required = required if required is not None else (default is None)
        if default is not None and required:
            raise ValueError(f'illegal argument values: required={required}; default={default}')

        # try to get the field value
        if field in dict_value:
            field_value = dict_value.pop(field)

            try:
                if validate_type is not None:
                    check_type(validate_type, field_value.value)  # type: ignore
                if validate:
                    field_value = ConfigValue.wrap(field_value.context, validate(field_value.unwrap()))
            except Exception as e:
                raise YamlParseError(self.context, f'illegal value for field `{field}`: {e}')

            return field_value

        # if no field value was returned, return the default value or raise an error
        if required:
            raise YamlParseError(self.context, f'missing field: `{field}`')
        elif default is not None:
            return ConfigValue.wrap(self.context + [field], default)
        else:
            return None

    def pop_field(
        self,
        field: str,
        default: Any = None,
        required: bool | None = None,
        validate: Callable[[Any], Any] | None = None,
        validate_type: type[Any] | None = None,
    ) -> Any:
        """
        Validate and retrieve the value for a given field, after which it is removed from the configuration.

        This method behaves similarly to ``pop()``, except that it returns an unwrapped value.

        Parameters
        ----------
        field : str
            Field name to retrieve.
        default : Any, optional
            Value to return when ``field`` is absent.
        required : bool | None, optional
            Whether to raise when the field is absent. Defaults to ``True`` when ``default`` is ``None``.
        validate : Callable[[Any], Any] | None, optional
            Validator function applied to the popped value.
        validate_type : type[Any] | None, optional
            Check that the popped value is of this type, or raise a ``ValueError``.

        Returns
        -------
        Any
            Popped field value or ``default``.
        """
        value = self.pop(field, default, required, validate, validate_type)
        return value.unwrap() if value is not None else None

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

    def as_dict(self, message: str | None = None) -> dict:  # numpydoc ignore=RT01
        """
        Return unwrapped dictionary or raise an error.

        If this ``ConfigValue`` is a dictionary, unwrap it and return the result. Otherwise, raise an error.

        Parameters
        ----------
        message : str | None
            A custom error message.
        """
        return check_type(dict, self.unwrap(), message=message)

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
        if isinstance(value, ConfigValue):
            raise ValueError(f'already wrapped: {value}')
        if isinstance(value, Mapping):
            return ConfigValue(context, {k: ConfigValue.wrap(context + [k], v) for k, v in value.items()})
        elif isinstance(value, str):
            return ConfigValue(context, value)
        elif isinstance(value, Sequence):
            return ConfigValue(context, [ConfigValue.wrap(context + [str(i)], value) for i, value in enumerate(value)])
        return ConfigValue(context, value)


class ConfigAttribute(NamedTuple):
    """
    An attribute in a configuration section.

    Attributes
    ----------
    name : str
        The attribute name.
    type : type[Any]
        The type of the attribute value.
    required : bool, optional
        Whether the attribute is required (defaults to ``False``).
    description : str, optional
        A text to describe the attribute and how it is used.
    """

    name: str
    type: type[Any] | UnionType
    required: bool = False
    description: str | None = None


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
    def _get_type_name(obj: Any) -> str:
        """
        Return the fully qualified type name of the ``obj`` type.

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
        Return the full class name that has the relevant docstring.

        By default, return the name of this class. In a subclass that was initialized with another class or function
        that does the actual work, the name of that class is returned.

        Returns
        -------
        str
            Fully qualified class name for this parser instance.
        """
        return self._get_type_name(self.__class__)

    def attributes(self) -> None | list[ConfigAttribute]:
        """
        Return the attributes in a configuration section that describes this object.

        Returns
        -------
        list[ConfigAttribute] | None
            The list of configuration attributes, or ``None`` if unknown.
        """
        return None


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
        return self._get_type_name(self.component_class)


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
        return self._get_type_name(self.component_class)


def get_full_name(obj: type[Any] | Callable) -> str:
    """
    Return the full name of an importable object.

    .. code-block:: python

        from lir import FeatureData
        print(get_full_name(FeatureData))
        'lir.data.models.FeatureData'

    This function does not yet handle type aliases.

    Parameters
    ----------
    obj : Any
        Importable object.

    Returns
    -------
    str
        Fully qualified object name.
    """
    if not hasattr(obj, '__name__'):
        raise ValueError(f'type object {obj} has no __name__ attribute; do you mean to call `type()` first?')

    return f'{obj.__module__}.{obj.__name__}'


def config_parser(
    func: Callable[[ConfigValue, Path], Any] | None = None,
    /,
    reference: str | Any | None = None,
    attributes: list[ConfigAttribute] | None = None,
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
    func : Callable[[ConfigValue, Path], Any], optional
        Function to wrap as a config parser.
    reference : str | Any, optional
        Explicit reference name or object used in generated metadata.
    attributes : list[ConfigAttribute], optional
        A list of attributes for the configuration parser.

    Returns
    -------
    Callable
        Decorator result or wrapped ``ConfigParser`` implementation.
    """
    if func is None:
        # take the optional arguments
        return partial(config_parser, reference=reference, attributes=attributes)

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

        def attributes(self) -> list[ConfigAttribute] | None:
            return attributes

    return ConfigParserFunction


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

    This is a legacy alternative for ``ConfigValue.pop()`` and ``ConfigValue.pop_field()``, and may be deprecated in the
    future.

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
    if unwrap is None:
        if default is not None and isinstance(default, ConfigValue):
            unwrap = False
        elif validate is not None or validate_type is not None or default is not None:
            unwrap = True
        else:
            unwrap = False

    if unwrap:
        return config.pop_field(
            field=field, default=default, required=required, validate=validate, validate_type=validate_type
        )
    else:
        return config.pop(
            field=field, default=default, required=required, validate=validate, validate_type=validate_type
        )


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
