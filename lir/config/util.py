from pathlib import Path
from typing import Any

from lir.config.base import ConfigParser, ConfigValue, pop_field
from lir.config.transform import parse_module
from lir.transform import Tee


class TeeParser(ConfigParser):
    """Parse configuration for allowing multiple tasks for given input."""

    def parse(
        self,
        config: ConfigValue,
        output_dir: Path,
    ) -> Any:
        """
        Read configuration for modules section and provide wrapped corresponding transformers.

        Parameters
        ----------
        config : ConfigValue
            Configuration for the `Tee` transformer, containing a `modules` field with a list of module configurations.
        output_dir : Path
            Output directory for the parsed modules.

        Returns
        -------
        Any
            A `Tee` transformer wrapping the parsed modules.
        """
        transformers = []
        modules = pop_field(config, 'modules')
        for module_config in modules:
            transformers.append(parse_module(module_config, output_dir))

        return Tee(transformers)
