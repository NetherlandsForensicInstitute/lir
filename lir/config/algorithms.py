from pathlib import Path
from typing import Any

from lir.algorithms.mcmc import McmcLLRModel
from lir.config.base import ContextAwareDict, config_parser, pop_field
from lir.config.transform import parse_module
from lir.util import partial


@config_parser
def mcmc(config: ContextAwareDict, output_dir: Path) -> McmcLLRModel:
    """
    Parse MCMC module configuration.

    Parameters
    ----------
    config : ContextAwareDict
        Configuration for the MCMC model.
    output_dir : Path
        Output directory used by nested parser calls.

    Returns
    -------
    McmcLLRModel
        Configured MCMC model instance.
    """
    if 'bounding' in config:
        bounding_config = pop_field(config, 'bounding')
        if bounding_config is None:
            config['bounding'] = None
        else:
            context = config.context + ['bounding'] if isinstance(bounding_config, str) else bounding_config.context
            bounding = partial(parse_module, bounding_config, output_dir, context)
            config['bounding'] = bounding

    mcmc_class: Any = McmcLLRModel
    return mcmc_class(**config)
