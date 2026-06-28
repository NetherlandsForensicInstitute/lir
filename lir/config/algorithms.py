from pathlib import Path
from typing import Any

from lir.algorithms.mcmc import McmcLLRModel
from lir.config.base import ConfigValue, config_parser, pop_field
from lir.config.transform import parse_module
from lir.util import partial


@config_parser
def mcmc(config: ConfigValue, output_dir: Path) -> McmcLLRModel:
    """
    Parse MCMC module configuration.

    Parameters
    ----------
    config : ConfigValue
        Configuration for the MCMC model.
    output_dir : Path
        Output directory used by nested parser calls.

    Returns
    -------
    McmcLLRModel
        Configured MCMC model instance.
    """
    bounding = None
    if 'bounding' in config and config['bounding'] is not None:
        bounding_config = pop_field(config, 'bounding')
        bounding = partial(parse_module, bounding_config, output_dir)

    mcmc_class: Any = McmcLLRModel
    return mcmc_class(**config.check_type(dict), bounding=bounding)
