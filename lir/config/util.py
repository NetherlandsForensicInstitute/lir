from pathlib import Path
from typing import Any

from lir import registry
from lir.config import ConfigValue, YamlParseError


def parse_config(
    config: ConfigValue,
    output_dir: Path,
    method_key: str,
    allow_shorthand: bool = False,
    default_method: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Parse a configuration section with a named parser.

    An error will be raised if the parser cannot be resolved or if the parsing fails.

    Parameters
    ----------
    config : ConfigValue
        Configuration section to parse.
    output_dir : Path
        The directory where output may be written.
    method_key : str
        The configuration key that refers to name of the parser that can be resolved by the registry.
    allow_shorthand : bool, optional
        Allow the shorthand form, meaning that ``config`` holds the object name as a ``str`` with no arguments.
    default_method : str, optional
        Default value for the method key field if it is not provided.
    **kwargs : Any
        Arguments forwarded to :meth:`~lir.registry.get`.

    Returns
    -------
    Any
        The parsed object.
    """
    if allow_shorthand and isinstance(config.value, str):
        object_name = config.value
        args = ConfigValue.wrap(config.context, {})
    else:
        object_name = config.pop_field(method_key, default=default_method, validate_type=str)
        args = config

    try:
        parser = registry.get(object_name, **kwargs)
    except registry.ComponentNotFoundError as e:
        raise YamlParseError(config.context, f'{e}')
    except Exception as e:
        raise YamlParseError(
            config.context,
            f'failed to load parser for {method_key} `{object_name}`; the error was: {e}',
        )

    return parser.parse(args, output_dir)
