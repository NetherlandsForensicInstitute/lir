from pathlib import Path
from typing import Any

from lir import registry
from lir.config.base import (
    YamlParseError,
    GenericConfigParser,
    pop_field,
)
from lir.data.models import DataSet


def parse_data_provider(cfg: dict[str, Any], context: list[str], output_path: Path) -> DataSet:
    """Instantiate specific implementation of `DataSetup` as configured.

    The `type` field is parsed, which is expected to refer to a name in
    the registry. See for example `lir.config.data_sources.synthesized_normal_binary`
    or `lir.config.data_sources.synthesized_normal_multiclass`.

    Data sources are provided under the `data_sources` key.
    """
    provider = pop_field(context, cfg, "provider")

    try:
        parser = registry.get(
            provider,
            search_path=["data_providers"],
            default_config_parser=GenericConfigParser,
        )
    except Exception as e:
        raise YamlParseError(
            context,
            f"no parser available for data provider `{provider}`; the error was: {e}",
        )

    return parser.parse(cfg, context, output_path)
