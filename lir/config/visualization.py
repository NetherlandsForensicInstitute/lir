from pathlib import Path
from typing import Callable, Any

from lir import registry
from lir.config.base import GenericFunctionConfigParser, YamlParseError, ContextAwareDict
from lir.registry import ComponentNotFoundError
from functools import partial
from collections.abc import Mapping, Sequence

def parse_visualizations(config: ContextAwareDict, output_path: Path) -> list[Callable[..., Any]]:
    """Prepare a list of functions to obtain the configured visualizations.

    Visualization functions must be available in the registry and accept three arguments:
       - output_dir: Path, the directory where output files can be created
       - lrs: np.ndarray, a list of lrs
       - labels: np.ndarray, a list of labels corresponding to the lrs
    """

    def _parse_params(module_cfg: Any, context: list[str]) -> dict[str, Any] | None:
        """Normalise module configuration into a parameter dict."""

        # No parameters provided
        if module_cfg is None:
            return None
        
        # Parameters provided as a dict/Mapping. In the yaml, it would look like:
        # visualization:
        #   - pav:
        #       h1_color: yellow
        #       h2_color: green
        if isinstance(module_cfg, Mapping):
            return dict(module_cfg)
        
        # Parameters provided as a list of single-entry dicts. In the yaml, it would look like:
        # visualization:
        #   - pav:
        #       - h1_color: yellow
        #       - h2_color: green
        if isinstance(module_cfg, Sequence):
            params = {}
            for el in module_cfg:
                params.update(el)
            return params
        
        # Invalid configuration format. This can happen if the user provides a single value, like
        # visualization:
        #   - pav: 
        # yellow
        raise YamlParseError(context, f"invalid visualization configuration: {module_cfg}")

    visualizations = []
    for entry in config:
        # Extract type and module config
        if isinstance(entry, str):
            visualization_type, module_cfg = entry, None
        else:
            visualization_type, module_cfg = next(iter(entry.items()))

        try:
            parser = registry.get(
                visualization_type,
                default_config_parser=GenericFunctionConfigParser,
                search_path=["visualization"],
            )
            func = parser.parse(ContextAwareDict(config.context + [visualization_type]), output_path)

            params = _parse_params(module_cfg, config.context)
            visualizations.append(partial(func, **params) if params else func)

        except ComponentNotFoundError as e:
            raise YamlParseError(config.context, str(e)) from e

    return visualizations

