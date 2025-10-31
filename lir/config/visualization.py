from collections import ChainMap
from pathlib import Path
from typing import Callable, Any

from lir import registry
from lir.config.base import GenericFunctionConfigParser, YamlParseError, ContextAwareDict
from lir.registry import ComponentNotFoundError
from functools import partial
from collections.abc import Mapping


def parse_visualizations(config: ContextAwareDict, output_path: Path) -> list[Callable[..., Any]]:
    """Prepare a list of functions to obtain the configured visualizations.

    Visualization functions must be available in the registry and accept three arguments:
       - output_dir: Path, the directory where output files can be created
       - lrs: np.ndarray, a list of lrs
       - labels: np.ndarray, a list of labels corresponding to the lrs

    Additionally, any provided extra options are parsed and passed as parameters to the
    corresponding visualization function. In the example below, the keyword arguments `h1_color`
    and `h2_color` will be passed to the corresponding function to generate `pav` plots.

    ```yaml
    visualization:
      - pav:
         - h1_color: yellow
         - h2_color: green
    ```
    """
    visualization_functions: list[Callable[..., Any]] = []

    # Collect the visualizations declared in the configuration as key value pairs of name/parameters
    defined_visualizations: dict[str, dict] = {}

    for visualization in config:
        # Collect visualization types and extra options, or use empty dictionary if no options were provided
        visualization_with_params = visualization if isinstance(visualization, Mapping) else {visualization: {}}
        defined_visualizations.update(visualization_with_params)

    for visualization_type, extra_options in defined_visualizations.items():
        try:
            parser = registry.get(
                visualization_type,
                default_config_parser=GenericFunctionConfigParser,
                search_path=["visualization"],
            )
        except ComponentNotFoundError as e:
            raise YamlParseError(config.context, str(e))

        visualization_fn = parser.parse(ContextAwareDict(config.context + [visualization_type]), output_path)

        # The `extra_options` are represented in a `ContextAwareList`, which needs to be casted to the
        # corresponding dictionary to provide the `**kwargs` to the visualization function.
        extra_parameters = dict(ChainMap(*extra_options))

        visualization_functions.append(partial(visualization_fn, **extra_parameters))

    return visualization_functions
