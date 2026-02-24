from pathlib import Path

import numpy as np
import numpy.random

from lir.config.base import ContextAwareDict, config_parser, pop_field
from lir.data.models import DataProvider, FeatureData


class SynthesizedNormalData:
    """Representation of normally distributed data, leveraging a number generator.

    The generated data can be used to generate normally distributed data and is useful
    for debugging purposes or gaining insight in the effect of varying parts within the
    LR system pipeline.
    """

    def __init__(self, mean: float, std: float, size: int | tuple[int, int]):
        self.mean = mean
        self.std = std
        # Convert size to a tuple if it is an integer, otherwise use the provided tuple.
        self.size = (size, 1) if isinstance(size, int) else size

    def get(self, rng: numpy.random.Generator) -> np.ndarray:
        """Draw random samples from a normally distributed data set."""
        return rng.normal(loc=self.mean, scale=self.std, size=self.size)


class SynthesizedNormalBinaryData(DataProvider):
    """Implementation of a data source generating normally distributed binary class data."""

    def __init__(self, h1_params: SynthesizedNormalData, h2_params: SynthesizedNormalData, seed: int | None = None):
        self.data_parameters = [h1_params, h2_params]
        self.seed = seed

    def get_instances(self) -> FeatureData:
        """
        Return instances with randomly synthesized data and binary labels.

        The features are drawn from a normal distribution, as configured.
        """
        rng = np.random.default_rng(seed=self.seed)
        features = np.concatenate([data_class.get(rng) for data_class in self.data_parameters])
        labels = np.concatenate([np.ones(self.data_parameters[0].size[0]), np.zeros(self.data_parameters[1].size[0])])
        return FeatureData(features=features, labels=labels)


@config_parser
def synthesized_normal_binary(config: ContextAwareDict, _: Path) -> DataProvider:
    """Set up (binary class) data source class to obtain normally distributed data from configuration."""
    seed = pop_field(config, 'seed', required=False)
    h1 = pop_field(config, 'h1')
    h2 = pop_field(config, 'h2')
    return SynthesizedNormalBinaryData(SynthesizedNormalData(**h2), SynthesizedNormalData(**h1), seed=seed)
