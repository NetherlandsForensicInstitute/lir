from collections.abc import Callable, Sequence
from pathlib import Path

from lir import LLRData, registry
from lir.config.base import ConfigValue, GenericFunctionConfigParser, YamlParseError, pop_field
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
        return parser.parse(ConfigValue.wrap(context, None), output_path)
    except ComponentNotFoundError as e:
        raise YamlParseError(context, str(e))


def parse_metric_mapping(
    config: ConfigValue, field: str, default: Sequence[str], output_path: Path
) -> dict[str, Callable[[LLRData], float]]:
    """
    Parse a list of metric names into a mapping of metric names to metric functions.

    Parameters
    ----------
    config : ConfigValue
        Configuration section containing the list of metric names.
    field : str
        Name of the configuration field that holds the list of metric names.
    default : Sequence[str]
        Default metric names to use when the field is not present.
    output_path : Path
        Output path passed to the metric parsers.

    Returns
    -------
    dict[str, Callable[[LLRData], float]]
        Mapping of metric names to metric callables.

    Raises
    ------
    YamlParseError
        If the configured value is not a list of metric names.
    """
    metric_names = pop_field(config, field, default=list(default))
    if not isinstance(metric_names, Sequence):
        raise YamlParseError(config.context, 'Invalid metrics configuration; expected a list of metric names.')

    return {name: parse_individual_metric(name, output_path, config.context) for name in metric_names}
