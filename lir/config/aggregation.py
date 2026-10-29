from pathlib import Path

from lir.aggregation import Aggregation
from lir.aggregation.group import AggregationGroup
from lir.config.base import (
    ConfigValue,
    GenericConfigParser,
    YamlParseError,
)
from lir.config.util import parse_config


def parse_aggregation(config: ConfigValue, output_dir: Path) -> Aggregation:
    """
    Parse a configuration section for output aggregation.

    If `config` is a dictionary, the `method` property is the aggregation method that is looked up in the registry.
    Other properties are passed as parameters. If `config` is a `str`, then its value is the aggregation method, and it
    has no parameters.

    Parameters
    ----------
    config : ConfigValue
        The configuration as a dictionary or string.
    output_dir : Path
        Output directory where derived artifacts are written.

    Returns
    -------
    Aggregation
        Parsed aggregation instance.
    """
    parsed_object = parse_config(
        config,
        output_dir,
        method_key='method',
        allow_shorthand=True,
        default_config_parser=GenericConfigParser,
        search_path=['output'],
    )

    if not isinstance(parsed_object, Aggregation):
        raise YamlParseError(
            config.context,
            f'Invalid output configuration; expected an Aggregation, found: {type(parsed_object)}.',
        )

    return parsed_object


def parse_aggregations(config: ConfigValue, output_dir: Path) -> Aggregation:
    """
    Parse a configuration section for an aggregation, or a list of aggregation configuration sections.

    Parameters
    ----------
    config : ConfigValue, optional
        Configuration for a single aggregation or a list of aggregation configurations.
    output_dir : Path
        Output directory for the aggregation instances.

    Returns
    -------
    Aggregation
        Parsed aggregation instance, or a group of aggregation instances.
    """
    if config.value is None:
        return AggregationGroup([])
    elif isinstance(config.value, list):
        return AggregationGroup([parse_aggregation(item, output_dir) for item in config.value])
    else:
        return parse_aggregation(config, output_dir)
