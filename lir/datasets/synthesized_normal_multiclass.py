from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from lir.config.base import config_parser, pop_field
from lir.config.substitution import ConfigValue
from lir.data.models import DataProvider, FeatureData


class SynthesizedDimension(NamedTuple):
    """Representation of a data distribution."""

    population_mean: float
    population_std: float
    sources_std: float


class SynthesizedNormalMulticlassData(DataProvider):
    """
    Implementation of a data source generating normally distributed multiclass data.

    Parameters
    ----------
    dimensions : list[SynthesizedDimension]
        Number of feature dimensions to include in the header.
    num_source_ids : int
        Number of sources to sample in the synthetic population.
    num_sources_per_source_id : int
        Number of source groups represented in the dataset.
    seed : int | None
        Random seed controlling stochastic behaviour for reproducible results.
    """

    def __init__(
        self,
        dimensions: list[SynthesizedDimension],
        num_source_ids: int,
        num_sources_per_source_id: int,
        seed: int | None,
    ):
        self.dimensions = dimensions
        self.num_source_ids = num_source_ids
        self.num_sources_per_source_id = num_sources_per_source_id
        self.seed = seed

    def _generate_dimension(self, rng: Any, dimension: SynthesizedDimension) -> np.ndarray:
        population = rng.normal(
            loc=dimension.population_mean,
            scale=dimension.population_std,
            size=self.num_source_ids,
        )
        measurement_error = rng.normal(
            loc=0,
            scale=dimension.sources_std,
            size=self.num_source_ids * self.num_sources_per_source_id,
        )
        measurements = np.concatenate([population] * self.num_sources_per_source_id) + measurement_error
        return measurements

    def get_instances(self) -> FeatureData:
        """
        Return instances with randomly synthesized data and multi-class labels.

        The features are drawn from a normal distribution, as configured.

        Returns
        -------
        FeatureData
            FeatureData object parsed from the source.
        """
        rng = np.random.default_rng(seed=self.seed)

        measurements = [self._generate_dimension(rng, dim) for dim in self.dimensions]
        measurements = np.stack(measurements, axis=1)
        source_ids = np.concatenate([np.arange(self.num_source_ids)] * self.num_sources_per_source_id)

        return FeatureData(features=measurements, source_ids=source_ids)


@config_parser
def synthesized_normal_multiclass(config: ConfigValue, _: Path) -> SynthesizedNormalMulticlassData:
    """
    Set up (multiple class) data source class to obtain normally distributed data from configuration.

    Parameters
    ----------
    config : ConfigValue
        Configuration mapping used to construct this component.
    _ : Path
        Unused argument required by the parser interface.

    Returns
    -------
    SynthesizedNormalMulticlassData
        Configured multiclass synthesized data provider.
    """
    seed = pop_field(config, 'seed', validate=int, required=False)

    population = pop_field(config, 'population')
    num_source_ids = pop_field(population, 'size', validate=int)
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
        num_source_ids,
        instances_per_source,
        seed,
    )
