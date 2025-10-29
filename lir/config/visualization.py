from pathlib import Path
from typing import Callable

from lir import registry
from lir.config.base import GenericFunctionConfigParser, YamlParseError, ContextAwareDict
from lir.registry import ComponentNotFoundError
from functools import partial
from collections.abc import Mapping


def parse_visualizations(config: ContextAwareDict, output_path: Path) -> list[Callable]:
    """Prepare a list of functions to obtain the configured visualizations.

    Visualization functions must be available in the registry and accept three arguments:
       - output_dir: Path, the directory where output files can be created
       - lrs: np.ndarray, a list of lrs
       - labels: np.ndarray, a list of labels corresponding to the lrs
    """
    visualization_functions = []
    for item in config:
        # support both simple list entries and dict entries
        # (e.g. `- pav: {h1_color: yellow, h2_color: green}` or just `- pav`)
        if isinstance(item, str):
            visualization_type = item
            module_cfg = None
        elif isinstance(item, Mapping):
            if len(item) != 1:
                raise YamlParseError(config.context, f"invalid visualization entry: {item}")
            visualization_type = next(iter(item.keys()))
            module_cfg = item[visualization_type]
        else:
            raise YamlParseError(config.context, f"invalid visualization entry type: {type(item)}")

        try:
            parser = registry.get(
                visualization_type,
                default_config_parser=GenericFunctionConfigParser,
                search_path=["visualization"],
            )
            func = parser.parse(ContextAwareDict(config.context + [visualization_type]), output_path)

            # If the config for this visualization contains parameters (e.g. h1_color/h2_color),
            # bind them as keyword arguments when calling the visualization function.
            if isinstance(module_cfg, ContextAwareDict) and len(module_cfg) > 0:
                params = dict(module_cfg)
                visualization_functions.append(partial(func, **params))
            else:
                visualization_functions.append(func)
        except ComponentNotFoundError as e:
            raise YamlParseError(config.context, str(e))

    return visualization_functions
