from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.random

from lir.config.base import ContextAwareDict, config_parser, pop_field
from lir.data.models import DataProvider
from lir.lrsystems.lrsystems import FeatureData


class SynthesizedNormalDataClass:
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

    def __init__(self, data_classes: Mapping[Any, SynthesizedNormalDataClass], seed: int):
        self.data_classes = data_classes
        self.seed = seed

    def get_instances(self) -> FeatureData:
        """
        Returns instances with randomly synthesized data and binary labels.

        The features are drawn from a normal distribution, as configured. The meta data vector is empty, with
        dimensions `(n, 0)`.
        """
        rng = np.random.default_rng(seed=self.seed)
        values = [(cls.get(rng), class_name) for class_name, cls in self.data_classes.items()]
        values = [(data, [class_name] * data.shape[0]) for data, class_name in values]
        features = np.concatenate([data for data, _ in values])
        labels = np.concatenate([labels for _, labels in values])
        return FeatureData(features=features, labels=labels)


@config_parser
def synthesized_normal_binary(config: ContextAwareDict, _: Path) -> DataProvider:
    """Set up (binary class) data source class to obtain normally distributed data from configuration."""
    seed = pop_field(config, 'seed')
    h1 = pop_field(config, 'h1')
    h2 = pop_field(config, 'h2')
    data_classes = {
        1: SynthesizedNormalDataClass(**h1),
        0: SynthesizedNormalDataClass(**h2),
    }
    return SynthesizedNormalBinaryData(data_classes, seed)
