from pathlib import Path
from typing import Callable, List

from confidence import Configuration

from lir import registry
from lir.config.base import GenericFunctionConfigParser, YamlParseError
from lir.registry import ComponentNotFoundError


def parse_visualizations(
    config: Configuration, context: List[str], output_path: Path
) -> List[Callable]:
    """Prepare a list of functions to obtain the configured visualizations.

    Visualization functions must be available in the registry and accept three arguments:
       - output_dir: Path, the directory where output files can be created
       - lrs: np.ndarray, a list of lrs
       - labels: np.ndarray, a list of labels corresponding to the lrs
    """
    visualization_functions = []
    for visualization_type in config:
        try:
            parser = registry.get(
                visualization_type,
                default_config_parser=GenericFunctionConfigParser,
                search_path=["visualization"],
            )
            func = parser.parse(config, context, output_path)
            visualization_functions.append(func)
        except ComponentNotFoundError as e:
            raise YamlParseError(context, str(e))

    return visualization_functions
