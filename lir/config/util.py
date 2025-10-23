from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from confidence import Configuration


from lir.config.base import ConfigParser, pop_field
from lir.config.transform import parse_module
from lir.transform import Tee


class TeeParser(ConfigParser):
    def parse(
        self,
        config: Mapping[str, Any],
        config_context_path: list[str],
        output_dir: Path,
    ) -> Any:
        transformers = []
        modules: Sequence[Configuration] = pop_field(config_context_path, config, "modules")
        for module_config in modules:
            transformers.append(parse_module(module_config, config_context_path, output_dir))

        return Tee(transformers)
