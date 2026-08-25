import functools
from collections.abc import Callable
from pathlib import Path

import numpy as np

from lir import LLRData
from lir.config import ConfigValue, config_parser, pop_field
from lir.util import logodds_to_odds


def calculate_weighted_cllr(llrdata: LLRData, h0_weight: float, h1_weight: float) -> float:
    lrs = logodds_to_odds(llrdata.llrs)
    lrs0 = lrs[llrdata.hypothesis == 0]
    lrs1 = lrs[llrdata.hypothesis == 1]
    cllr0 = h0_weight * np.mean(np.log2(1 + lrs0))
    cllr1 = h1_weight * np.mean(np.log2(1 + 1 / lrs1))
    return (cllr0 + cllr1) / (h0_weight + h1_weight)


@config_parser
def weighted_cllr(config: ConfigValue, output_dir: Path) -> Callable[[LLRData], float]:
    h0_weight = pop_field(config, 'h0_weight', validate=float, default=1.0)
    h1_weight = pop_field(config, 'h1_weight', validate=float, default=1.0)
    return functools.partial(calculate_weighted_cllr, h0_weight=h0_weight, h1_weight=h1_weight)
