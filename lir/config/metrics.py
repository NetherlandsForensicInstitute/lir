from collections.abc import Callable
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

        metric = parse_individual_metric(metric_config, output_path)
        if name in metrics:
            raise YamlParseError(metric_config.context, f'duplicate metric name: `{name}`')
        metrics[name] = metric

    return metrics
