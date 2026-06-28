from collections.abc import Callable
from pathlib import Path

from lir import LLRData, registry
from lir.config.base import ConfigValue, GenericFunctionConfigParser, YamlParseError
from lir.registry import ComponentNotFoundError


def parse_individual_metric(name: str, output_path: Path, context: list[str]) -> Callable[[LLRData], float]:
    """
    Parse one metric from the registry.

    Parameters
    ----------
    name : str
        Registered metric name.
    output_path : Path
        Output path passed to the metric parser.
    context : list[str]
        YAML context used for error reporting.

    Returns
    -------
    Callable
        Metric callable.
    """
    try:
        parser = registry.get(
            name,
            default_config_parser=GenericFunctionConfigParser,
            search_path=['metric'],
        )
        return parser.parse(ConfigValue(context, None), output_path)
    except ComponentNotFoundError as e:
        raise YamlParseError(context, str(e))
