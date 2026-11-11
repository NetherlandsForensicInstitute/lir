from collections.abc import Callable, Sequence
from pathlib import Path

from lir import LLRData
from lir.config.base import ConfigValue, GenericFunctionConfigParser, YamlParseError
from lir.config.util import parse_config


def parse_individual_metric(config: ConfigValue, output_path: Path) -> Callable[[LLRData], float]:
    """
    Parse one metric from the registry.

    A metric is either the registered name of the metric, or a mapping with the registered name in the ``method`` field
    and the arguments of the metric in the remaining fields.

    Parameters
    ----------
    config : ConfigValue
        Registered metric name, or metric configuration section.
    output_path : Path
        Output path passed to the metric parser.

    Returns
    -------
    Callable
        Metric callable.
    """
    return parse_config(
        config,
        output_path,
        method_key='method',
        allow_shorthand=True,
        default_config_parser=GenericFunctionConfigParser,
        search_path=['metric'],
    )


def parse_metrics(config: ConfigValue, output_path: Path) -> dict[str, Callable[[LLRData], float]]:
    """
    Parse a list of metric definitions into a mapping of names to metric callables.

    Each entry is either the registered name of a metric or a mapping with the registered name in the ``method`` field.
    The name of the metric is used as the key in the resulting mapping, unless the ``name`` field overrides it. This key
    is used to label the metric in the output. Example:

    .. code-block:: yaml

        columns:
          - cllr
          - name: cllr_forensic
            method: cllr
            h1_weight: 0.9

    Parameters
    ----------
    config : ConfigValue
        List of metric names or metric configuration sections.
    output_path : Path
        Output path passed to the metric parsers.

    Returns
    -------
    dict[str, Callable]
        Mapping of metric names to metric callables.

    Raises
    ------
    YamlParseError
        If a metric definition is invalid, or if two metrics share the same name.
    """
    if not isinstance(config.value, list):
        raise YamlParseError(config.context, f'expected a list of metrics; found type: {type(config.value)}')

    metrics: dict[str, Callable[[LLRData], float]] = {}
    for metric_config in config.value:
        name = (
            metric_config.pop_field('name', required=False, validate_type=str) or str(metric_config['method'].value)
            if isinstance(metric_config.value, dict) and 'method' in metric_config
            else str(metric_config.value)
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
