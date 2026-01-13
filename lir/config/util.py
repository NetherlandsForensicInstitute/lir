from pathlib import Path
from typing import Any

from lir.config.base import ConfigParser, ContextAwareDict, pop_field
from lir.config.transform import parse_module
from lir.transform import Tee


class TeeParser(ConfigParser):
    def parse(
        self,
        config: ContextAwareDict,
        output_dir: Path,
    ) -> Any:
        transformers = []
        modules = pop_field(config, 'modules')
        for module_config in modules:
            transformers.append(parse_module(module_config, output_dir, module_config.context))

        return Tee(transformers)


def simplify_data_structure(data: Any) -> Any:
    """
    Simplify data structure: specialized data types are replaced.

    For example, `ContextAwareDict` is replaced by `dict`.
    """
    match data:
        case dict():
            return {k: simplify_data_structure(v) for k, v in data.items()}
        case list() | tuple():
            return [simplify_data_structure(v) for v in data]
        case str() | float() | int() | bool() | None:
            return data
        case _:
            raise ValueError(f'unrecognized data type: {data} of type {type(data)}')
