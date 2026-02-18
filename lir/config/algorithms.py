from pathlib import Path
from typing import Any

from lir.algorithms.mcmc import McmcLLRModel
from lir.bounding import LLRBounder
from lir.config.base import ContextAwareDict, YamlParseError, config_parser, pop_field
from lir.config.transform import parse_module
from lir.transform import TransformerWrapper


def _unwrap_bounder(bounder: object, context: list[str]) -> LLRBounder:
    """Unwrap transformer wrappers and validate that the result is an `LLRBounder`."""
    while isinstance(bounder, TransformerWrapper):
        bounder = bounder.wrapped_transformer

    if not isinstance(bounder, LLRBounder):
        raise YamlParseError(
            context,
            f'`bounding` should resolve to an LLRBounder; found: {type(bounder).__name__}',
        )

    return bounder


@config_parser
def mcmc(config: ContextAwareDict, output_dir: Path) -> McmcLLRModel:
    """Parse MCMC module config and resolve optional `bounding` to an `LLRBounder` instance."""
    if 'bounding' in config:
        bounding_config = pop_field(config, 'bounding')
        if bounding_config is None:
            config['bounding'] = None
        else:
            context = config.context + ['bounding'] if isinstance(bounding_config, str) else bounding_config.context
            bounder = parse_module(bounding_config, output_dir, context)
            config['bounding'] = _unwrap_bounder(bounder, context)

    mcmc_class: Any = McmcLLRModel
    return mcmc_class(**config)
