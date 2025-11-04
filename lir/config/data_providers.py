from pathlib import Path

from lir import registry
from lir.config.base import (
    GenericConfigParser,
    YamlParseError,
    pop_field,
)
from lir.config.substitution import ContextAwareDict
from lir.data.models import DataSet


def parse_data_provider(cfg: ContextAwareDict, output_path: Path) -> DataSet:
    """Instantiate specific implementation of `DataSetup` as configured.

    The `type` field is parsed, which is expected to refer to a name in
    the registry. See for example `lir.config.data_sources.synthesized_normal_binary`
    or `lir.config.data_sources.synthesized_normal_multiclass`.

    Data sources are provided under the `data_sources` key.
    """
    provider = pop_field(cfg, 'provider')

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
