from pathlib import Path
from typing import Any, NamedTuple, Optional

import numpy as np

from lir.config.base import config_parser, pop_field
from lir.data.models import DataSet


class SynthesizedDimension(NamedTuple):
    population_mean: float
    population_std: float
    sources_std: float


class SynthesizedNormalMulticlassData(DataSet):
    """Implementation of a data source generating normally distributed multiclass data."""

    def __init__(
        self,
        dimensions: list[SynthesizedDimension],
        population_size: int,
        sources_size: int,
        seed: Optional[int],
    ):
        self.dimensions = dimensions
        self.population_size = population_size
        self.sources_size = sources_size
        self.seed = seed

    def _generate_dimension(
        self, rng: Any, dimension: SynthesizedDimension
    ) -> np.ndarray:
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
        measurements = (
            np.concatenate([population] * self.sources_size) + measurement_error
        )
        return measurements

    def get_instances(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return instances with randomly synthesized data and multi-class labels.

        The features are drawn from a normal distribution, as configured. The meta data vector contains, for each
        instance, the "true" values for its source, without the error that is included in the feature values.
        """
        rng = np.random.default_rng(seed=self.seed)

        measurements = [self._generate_dimension(rng, dim) for dim in self.dimensions]
        measurements = np.stack(measurements, axis=1)
        labels = np.concatenate([np.arange(self.population_size)] * self.sources_size)
        meta = np.zeros((measurements.shape[0], 0))

        return measurements, labels, meta


@config_parser
def synthesized_normal_multiclass(
    config: dict[str, Any], config_context_path: list[str], _: Path
) -> DataSet:
    """Set up (multiple class) data source class to obtain normally distributed data from configuration."""
    seed = pop_field(config_context_path, config, "seed", validate=int, required=False)

    population = pop_field(config_context_path, config, "population")
    population_size = pop_field(
        config_context_path + ["population"], population, "size", validate=int
    )
    instances_per_source = pop_field(
        config_context_path + ["population"],
        population,
        "instances_per_source",
        validate=int,
    )

    dimensions_cfg = pop_field(config_context_path, config, "dimensions")
    dimensions = []
    for i, dim in enumerate(dimensions_cfg):
        dimension_context = config_context_path + ["dimensions", str(i)]
        mean = pop_field(dimension_context, dim, "mean", validate=float)
        std = pop_field(dimension_context, dim, "std", validate=float)
        error_std = pop_field(dimension_context, dim, "error_std", validate=float)
        dimensions.append(SynthesizedDimension(mean, std, error_std))

    return SynthesizedNormalMulticlassData(
        dimensions,
        population_size,
        instances_per_source,
        seed,
    )
