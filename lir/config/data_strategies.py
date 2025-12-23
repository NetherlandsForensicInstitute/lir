from pathlib import Path

from lir import registry
from lir.config.base import (
    GenericConfigParser,
    YamlParseError,
    pop_field,
)
from lir.config.substitution import ContextAwareDict
from lir.data.models import DataStrategy


def parse_data_strategy(cfg: ContextAwareDict, output_path: Path) -> DataStrategy:
    """Instantiate specific implementation of `DataStrategy` as configured.

    The `setup` field is parsed, which is expected to refer to a name in
    the registry. See for example `lir.data_setup.binary_cross_validation`
    or `lir.data_setup.binary_train_test_split`.

    Data setup configuration is provided under the `data_setup` key.
    """
    strategy = pop_field(cfg, 'strategy')

    try:
        parser = registry.get(
            strategy,
            search_path=['data_strategies'],
            default_config_parser=GenericConfigParser,
        )
    except Exception as e:
        raise YamlParseError(
            cfg.context,
            f'no parser available for data type `{strategy}`; the error was: {e}',
        )

    return parser.parse(cfg, output_path)
