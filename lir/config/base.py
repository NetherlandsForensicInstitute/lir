import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

from lir import registry
from lir.transform.pairing import PairingMethod


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
        """
        Initialise the parse error.

        Parameters
        ----------
        config_context_path : list[str]
            Dot-path to the failing configuration node.
        message : str
            Human-readable validation or parsing message.
        """
        prefix = f'{".".join(config_context_path)}: ' if config_context_path else ''
        super().__init__(f'{prefix}{message}')


class ContextAwareDict(dict):
    """
    Dictionary wrapper which has knowledge about its context.

    Parameters
    ----------
    context : list[str]
        YAML path used for contextual error messages.
    *args : Any
        Positional arguments passed to ``dict``.
    **kwargs : Any
        Keyword arguments passed to ``dict``.
    """

    def __init__(self, context: list[str], *args: Any, **kwargs: Any):
        """
        Create a context-aware dictionary.

        Parameters
        ----------
        context : list[str]
            YAML path used for contextual error messages.
        *args : Any
            Positional arguments passed to ``dict``.
        **kwargs : Any
            Keyword arguments passed to ``dict``.
        """
        super().__init__(*args, **kwargs)
        self.context = context

    def clone(self, context: list[str] | None = None) -> 'ContextAwareDict':
        """
        Create a cloned dictionary with expanded nested context.

        Parameters
        ----------
        context : list[str] | None, optional
            Replacement context. If omitted, the current context is reused.

        Returns
        -------
        ContextAwareDict
            Cloned and context-aware dictionary.
        """
        return _expand(context if context is not None else self.context, self)


class ContextAwareList(list):
    """
    List wrapper which has knowledge about its context.

    Parameters
    ----------
    context : list[str]
        YAML path used for contextual error messages.
    *args : Any
        Positional arguments passed to ``list``.
    **kwargs : Any
        Keyword arguments passed to ``list``.
    """

    def __init__(self, context: list[str], *args: Any, **kwargs: Any):
        """
        Create a context-aware list.

        Parameters
        ----------
        context : list[str]
            YAML path used for contextual error messages.
        *args : Any
            Positional arguments passed to ``list``.
        **kwargs : Any
            Keyword arguments passed to ``list``.
        """
        super().__init__(*args, **kwargs)
        self.context = context

    def clone(self, context: list[str] | None = None) -> 'ContextAwareList':
        """
        Create a cloned list with expanded nested context.

        Parameters
        ----------
        context : list[str] | None, optional
            Replacement context. If omitted, the current context is reused.

        Returns
        -------
        ContextAwareList
            Cloned and context-aware list.
        """
        return _expand(context if context is not None else self.context, self)


def _expand(context: list[str], cfg: Any) -> Any:
    """
    Expand nested values into context-aware containers.

    Parameters
    ----------
    context : list[str]
        Current YAML path.
    cfg : Any
        Value to expand recursively.

    Returns
    -------
    Any
        Expanded value where mappings and sequences are wrapped in context-aware types.
    """
    if isinstance(cfg, Mapping):
        return ContextAwareDict(context, [(key, _expand(context + [key], value)) for key, value in cfg.items()])
    elif isinstance(cfg, str):
        return cfg
    elif isinstance(cfg, Sequence):
        return ContextAwareList(context, [_expand(context + [str(i)], value) for i, value in enumerate(cfg)])
    return cfg


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
        config: ContextAwareDict,
        output_dir: Path,
    ) -> Any:
        """
        Parse a specific configuration section.

        Parameters
        ----------
        config : ContextAwareDict
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
        """
        Initialise the parser.

        Parameters
        ----------
        component_class : Callable
            Callable that should be exposed by this parser.
        """
        super().__init__()
        self.component_class = component_class

    def parse(
        self,
        config: ContextAwareDict,
        output_dir: Path,
    ) -> Callable:
        """
        Parse configuration into a callable.

        Parameters
        ----------
        config : ContextAwareDict
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
        """
        Initialise the parser.

        Parameters
        ----------
        component_class : type[Any]
            Class to instantiate from configuration values.
        """
        super().__init__()
        self.component_class = component_class

    def parse(
        self,
        config: ContextAwareDict,
        output_dir: Path,
    ) -> Any:
        """
        Instantiate the configured component class.

        Parameters
        ----------
        config : ContextAwareDict
            Keyword arguments for class initialisation.
        output_dir : Path
            Unused output directory argument required by the parser API.

        Returns
        -------
        Any
            Instantiated object.
        """
        try:
            return self.component_class(**config)
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
    func: Callable[[ContextAwareDict, Path], Any] | None = None, /, reference: str | Any | None = None
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

    Parameters
    ----------
    func : Callable[[ContextAwareDict, Path], Any] | None, optional
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
            config: ContextAwareDict,
            output_dir: Path,
        ) -> Any:
            return func(config, output_dir)  # type: ignore

        def reference(self) -> str:
            # return the reference argument, if any
            if reference is not None:
                return reference if isinstance(reference, str) else get_full_name(reference)

            assert func is not None  # at this point, func is always available

            # return the return type of the function, if available
            return_type = inspect.signature(func).return_annotation
            if not isinstance(return_type, str):
                return get_full_name(return_type)

            # last resort: fallback to wrapped function name
            return get_full_name(func)

    return ConfigParserFunction


def parse_pairing_config(
    module_config: ContextAwareDict | str,
    output_dir: Path,
    context: list[str],
) -> PairingMethod:
    """
    Parse and delegate pairing to the corresponding function for the defined pairing method.

    The argument `module_config` defines the pairing method. If its value is a `str`, the registry is queried and the
    corresponding pairing method is returned. If its value is a `dict`, the pairing method is defined
    by the value `module_config["method"]`, and the registry is queried for the config parser of
    the corresponding pairing method. The remaining values in `module_config` are passed as arguments to the
    configuration parser of the pairing method.

    If the registry cannot resolve the pairing method, an exception is raised.

    Parameters
    ----------
    module_config : ContextAwareDict | str
        Pairing method configuration.
    output_dir : Path
        Output directory for parser calls.
    context : list[str]
        Context used when ``module_config`` is a string.

    Returns
    -------
    PairingMethod
        Parsed pairing method.
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

    Parameters
    ----------
    type_class : Any
        Target type or tuple of target types.
    v : YamlValueType
        Value to validate.
    message : str | None, optional
        Error message used when validation fails.

    Returns
    -------
    Any
        Original value when type validation succeeds.
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

    Parameters
    ----------
    config : ContextAwareDict | Any
        Configuration object to pop from.
    field : str
        Field name to retrieve.
    default : Any, optional
        Value to return when ``field`` is absent.
    required : bool | None, optional
        Whether to raise when the field is absent. Defaults to ``True`` when
        ``default`` is ``None``.
    validate : Callable[[Any], Any] | None, optional
        Optional validator applied to the popped value.

    Returns
    -------
    Any
        Popped field value or ``default``.
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
    """
    Ensure all defined expected arguments are parsed and warn about ignored arguments.

    If any unexpected arguments remain, a `YamlParseError` is raised indicating the
    argument was unexpected and not taken into account (i.e. not parsed). This methodology ensures
    the user does not assume arguments are parsed that are in fact not recognized.

    Parameters
    ----------
    config : ContextAwareDict
        Configuration to validate for remaining keys.
    accept_keys : Sequence[str] | None, optional
        Keys that may remain without raising an error.

    Returns
    -------
    None
        This function raises on invalid input and otherwise returns ``None``.
    """
    for key in config:
        if not accept_keys or key not in accept_keys:
            raise YamlParseError(config.context, f'unrecognized argument: {key}')
