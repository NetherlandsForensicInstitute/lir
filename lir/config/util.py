from pathlib import Path
from typing import Any


from lir.config.base import ConfigParser, pop_field, ContextAwareDict
from lir.config.transform import parse_module
from lir.transform import Tee


class TeeParser(ConfigParser):
    def parse(
        self,
        config: ContextAwareDict,
        output_dir: Path,
    ) -> Any:
        transformers = []
        modules = pop_field(config, "modules")
        for module_config in modules:
            transformers.append(parse_module(module_config, output_dir, module_config.context))

        return Tee(transformers)
