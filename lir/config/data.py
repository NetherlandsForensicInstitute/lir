from pathlib import Path

from lir import registry
from lir.config.base import (
    GenericConfigParser,
    YamlParseError,
    pop_field,
)
from lir.config.substitution import ContextAwareDict
from lir.data.models import DataProvider, DataStrategy


def parse_data_object(cfg: ContextAwareDict, output_path: Path) -> tuple[DataProvider, DataStrategy]:
    """Parse data provider and data strategy from configuration.

    The `provider` and `splits` fields are parsed, which are expected to refer
    to specific implementations of `DataProvider` and `DataStrategy`, respectively.
    See `parse_data_provider` and `parse_data_strategy` for more information.
    """
    return parse_data_provider(cfg['provider'], output_path), parse_data_strategy(cfg['splits'], output_path)


def parse_data_strategy(cfg: ContextAwareDict, output_path: Path) -> DataStrategy:
    """Instantiate specific implementation of `DataStrategy` as configured.

    The `strategy` field is parsed, which is expected to refer to a name in
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
            f'no parser available for data strategy `{strategy}`; the error was: {e}',
        )

    return parser.parse(cfg, output_path)


def parse_data_provider(cfg: ContextAwareDict, output_path: Path) -> DataProvider:
    """Instantiate specific implementation of `DataProvider` as configured.

    The `method` field is parsed, which is expected to refer to a name in
    the registry. See for example `lir.config.data_sources.synthesized_normal_binary`
    or `lir.config.data_sources.synthesized_normal_multiclass`.

    Data sources are provided under the `data_sources` key.
    """
    provider = pop_field(cfg, 'method')

    try:
        parser = registry.get(
            provider,
            search_path=['data_providers'],
            default_config_parser=GenericConfigParser,
        )
    except Exception as e:
        raise YamlParseError(
            cfg.context,
            f'no parser available for data provider `{provider}`; the error was: {e}',
        )

    return parser.parse(cfg, output_path)
