from collections.abc import Callable
from pathlib import Path

from lir import registry
from lir.config.base import ContextAwareDict, GenericFunctionConfigParser, YamlParseError
from lir.registry import ComponentNotFoundError


def parse_individual_metric(name: str, output_path: Path, context: list[str]) -> Callable:
    try:
        parser = registry.get(
            name,
            default_config_parser=GenericFunctionConfigParser,
            search_path=['metric'],
        )
        return parser.parse(ContextAwareDict(context), output_path)
    except ComponentNotFoundError as e:
        raise YamlParseError(context, str(e))
