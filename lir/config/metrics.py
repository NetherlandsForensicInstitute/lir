from pathlib import Path

from lir import registry
from lir.config.base import ContextAwareDict, GenericFunctionConfigParser, YamlParseError
from lir.metrics.base import MetricFunction
from lir.registry import ComponentNotFoundError


def parse_individual_metric(name: str, output_path: Path, context: list[str]) -> MetricFunction:
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
    MetricFunction
        A metric function object.
    """
    try:
        parser = registry.get(
            name,
            default_config_parser=GenericFunctionConfigParser,
            search_path=['metric'],
        )
        return parser.parse(ContextAwareDict(context), output_path)
    except ComponentNotFoundError as e:
        raise YamlParseError(context, str(e))
