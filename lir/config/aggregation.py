from functools import partial
from pathlib import Path

from lir import registry
from lir.aggregation import Aggregation, SubsetAggregation
from lir.config.base import (
    ConfigParser,
    ContextAwareDict,
    ContextAwareList,
    GenericConfigParser,
    YamlParseError,
    check_type,
    config_parser,
    pop_field,
)


def parse_aggregation(
    config: ContextAwareDict | str, output_dir: Path, context: list[str] | None = None
) -> Aggregation:
    """
    Parse a configuration section for output aggregation.

    If `config` is a dictionary, the `method` property is the aggregation method that is looked up in the registry.
    Other properties are passed as parameters. If `config` is a `str`, then its value is the aggregation method, and it
    has no parameters.

    :param config: the configuration as a dictionary or str
    :param output_dir: the output directory
    :param context: the context cof the configuration (if `config` is a str)
    :return: an Aggregation object
    """
    # Normalise configuration into (class_name, args)
    if isinstance(config, str):
        config = ContextAwareDict(context or [], {'method': config})
    check_type(dict, config, 'invalid output configuration; expected a string or a mapping with a "method" field')

    class_name = pop_field(config, 'method', validate=partial(check_type, str))

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


def parse_aggregations(
    config: ContextAwareList | ContextAwareDict | str, output_dir: Path, context: list[str] | None = None
) -> list[Aggregation]:
    """
    Parse a list of configurations for aggregation.

    :param config: the configuration section of a single aggregation, or a list of aggregation configurations
    :param output_dir: the output directory for the aggregations
    :param context: the context cof the configuration (if `config` is a `str`)
    :return: a list of Aggregation objects
    """
    context = context or getattr(config, 'context', None) or []
    if isinstance(config, list):
        return [parse_aggregation(item, output_dir, context) for i, item in enumerate(config)]
    else:
        return [parse_aggregation(config, output_dir, context)]


@config_parser
def subset_aggregation(config: ContextAwareDict, output_dir: Path) -> SubsetAggregation:
    """
    Parse a configuration section for a categorized subset aggregation.

    :param config: the configuration section
    :param output_dir: the output directory
    :return: a SubsetAggregation object
    """
    check_type(dict, config, 'output configuration should be a dictionary')
    category_field = pop_field(config, 'category_field')
    subset_output_dir = output_dir / category_field

    aggregation_config = pop_field(config, 'output')
    if isinstance(aggregation_config, list):
        aggregation_methods = [parse_aggregation(item, subset_output_dir) for item in aggregation_config]
    else:
        aggregation_methods = [parse_aggregation(aggregation_config, subset_output_dir)]

    return SubsetAggregation(aggregation_methods, category_field)
