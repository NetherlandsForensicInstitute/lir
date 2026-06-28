from pathlib import Path

from lir import registry
from lir.aggregation import Aggregation, SubsetAggregation
from lir.config.base import (
    ConfigParser,
    ConfigValue,
    GenericConfigParser,
    YamlParseError,
    config_parser,
    pop_field,
)


def parse_aggregation(config: ConfigValue, output_dir: Path, context: list[str] | None = None) -> Aggregation:
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
        Output directory where derived artefacts are written.
    context : list[str] | None, optional
        Context for error reporting when ``config`` is provided as a string.

    Returns
    -------
    Aggregation
        Parsed aggregation instance.
    """
    # Normalise configuration into (class_name, args)
    if isinstance(config.value, str):
        class_name = config.value
        config = ConfigValue(config.context, {})
    else:
        config.check_type(
            dict, message='invalid output configuration; expected a string or a mapping with a "method" field'
        )
        class_name = pop_field(config, 'method', validate_type=str)

    parser: ConfigParser = registry.get(
        class_name,
        default_config_parser=GenericConfigParser,
        search_path=['output'],
    )
    parsed_object = parser.parse(config, output_dir)

    if not isinstance(parsed_object, Aggregation):
        raise YamlParseError(
            config.context,
            f'Invalid output configuration; expected an Aggregation, found: {type(parsed_object)}.',
        )

    return parsed_object


def parse_aggregations(config: ConfigValue, output_dir: Path) -> list[Aggregation]:
    """
    Parse a list of configurations for aggregation.

    Parameters
    ----------
    config : ConfigValue
        Configuration for a single aggregation or a list of aggregation configurations.
    output_dir : Path
        Output directory for the aggregation instances.

    Returns
    -------
    list[Aggregation]
        Parsed aggregation instances.
    """
    if isinstance(config.value, list):
        return [parse_aggregation(item, output_dir) for i, item in enumerate(config.value)]
    else:
        return [parse_aggregation(config, output_dir)]


@config_parser
def subset_aggregation(config: ConfigValue, output_dir: Path) -> SubsetAggregation:
    """
    Parse a configuration section for a categorized subset aggregation.

    Parameters
    ----------
    config : ConfigValue
        Configuration section.
    output_dir : Path
        Output directory.

    Returns
    -------
    SubsetAggregation
        Parsed subset aggregation object.
    """
    config.check_type(dict, message='output configuration should be a dictionary')
    category_field = pop_field(config, 'category_field', validate=str)
    subset_output_dir = output_dir / category_field

    aggregation_config = pop_field(config, 'output')
    if isinstance(aggregation_config, list):
        aggregation_methods = [parse_aggregation(item, subset_output_dir) for item in aggregation_config]
    else:
        aggregation_methods = [parse_aggregation(aggregation_config, subset_output_dir)]

    return SubsetAggregation(aggregation_methods, category_field)
