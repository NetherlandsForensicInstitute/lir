from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from lir.config.base import config_parser, pop_field
from lir.config.substitution import ContextAwareDict
from lir.data.models import DataProvider
from lir.lrsystems.lrsystems import FeatureData


class SynthesizedDimension(NamedTuple):
    population_mean: float
    population_std: float
    sources_std: float


class SynthesizedNormalMulticlassData(DataProvider):
    """Implementation of a data source generating normally distributed multiclass data."""

    def __init__(
        self,
        dimensions: list[SynthesizedDimension],
        population_size: int,
        sources_size: int,
        seed: int | None,
    ):
        self.dimensions = dimensions
        self.population_size = population_size
        self.sources_size = sources_size
        self.seed = seed

    def _generate_dimension(self, rng: Any, dimension: SynthesizedDimension) -> np.ndarray:
        population = rng.normal(
            loc=dimension.population_mean,
            scale=dimension.population_std,
            size=self.population_size,
        )
        measurement_error = rng.normal(
            loc=0,
            scale=dimension.sources_std,
            size=self.population_size * self.sources_size,
        )
        measurements = np.concatenate([population] * self.sources_size) + measurement_error
        return measurements

    def get_instances(self) -> FeatureData:
        """
        Return instances with randomly synthesized data and multi-class labels.

        The features are drawn from a normal distribution, as configured.
        """
        rng = np.random.default_rng(seed=self.seed)

        measurements = [self._generate_dimension(rng, dim) for dim in self.dimensions]
        measurements = np.stack(measurements, axis=1)
        source_ids = np.concatenate([np.arange(self.population_size)] * self.sources_size)

        return FeatureData(features=measurements, source_ids=source_ids)


@config_parser
def synthesized_normal_multiclass(config: ContextAwareDict, _: Path) -> DataProvider:
    """Set up (multiple class) data source class to obtain normally distributed data from configuration."""
    seed = pop_field(config, 'seed', validate=int, required=False)

    population = pop_field(config, 'population')
    population_size = pop_field(population, 'size', validate=int)
    instances_per_source = pop_field(
        population,
        'instances_per_source',
        validate=int,
    )

    dimensions_cfg = pop_field(config, 'dimensions')
    dimensions = []
    for dim in dimensions_cfg:
        mean = pop_field(dim, 'mean', validate=float)
        std = pop_field(dim, 'std', validate=float)
        error_std = pop_field(dim, 'error_std', validate=float)
        dimensions.append(SynthesizedDimension(mean, std, error_std))

    return SynthesizedNormalMulticlassData(
        dimensions,
        population_size,
        instances_per_source,
        seed,
    )
