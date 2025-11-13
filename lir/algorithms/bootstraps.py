from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

import numpy as np
from scipy.interpolate import interp1d

from lir.config.base import ContextAwareDict, check_not_none, config_parser, pop_field
from lir.data.models import FeatureData, LLRData
from lir.transform.pipeline import Pipeline, parse_steps


class Bootstrap(Pipeline, ABC):
    """Bootstrap system that estimates confidence intervals around the best estimate of a pipeline.

    This bootstrap system creates bootstrap samples from the training data, fits the pipeline on each sample,
    and then computes confidence intervals for the pipeline outputs based on the variability across the bootstrap
    samples.

    Computing these intervals is done by creating interpolation functions that map the best estimate to the
    difference between the best estimate and the lower and upper bounds of the confidence interval. To achieve this,
    two subclasses are provided that differ in how the data points for interval estimation are obtained.

    - BootstrapAtData: Uses the original training data points for interval estimation.
    - BootstrapEquidistant: Uses equidistant points within the range of the training data for interval estimation.

    The AtData variant allows for more complex data types, while the Equidistant variant is only suitable for continuous
    features.

    Attributes:
        steps: The steps of s pipeline to be bootstrapped.
        n_bootstraps: int: The number of bootstrap samples to generate.
        interval: tuple[float, float]: The lower and upper quantiles for the confidence interval.
        seed: int | None: The random seed for reproducibility.
        n_points: int | None: Number of equidistant points to use for interval estimation (BootstrapEquidistant only).

        f_delta_interval_lower: Interpolation function for the lower bound of the interval.
        f_delta_interval_upper: Interpolation function for the upper bound of the interval.
    """

    def __init__(
        self,
        steps: list[tuple[str, Any]],
        n_bootstraps: int = 400,
        interval: tuple[float, float] = (0.05, 0.95),
        seed: int | None = None,
    ):
        """Initialize the TrainDataBootstrap with the given pipeline steps, number of bootstraps, and interval.

        Parameters:
        param steps: list[tuple[str, Any]]: The steps of the pipeline to be bootstrapped.
        param n_bootstraps: int: The number of bootstrap samples to generate. Default is 400.
        param interval: tuple[float, float]: The lower and upper quantiles for the confidence interval.
                                             Default: (0.05,0.95).
        param n_points: int | None: The number of equidistant points to use for interval estimation. Default is 1000.
                                    If None, uses the number of instances in the training data.
        """
        super().__init__(steps)

        self.interval = interval
        self.n_bootstraps = n_bootstraps
        self.seed = seed

        self.f_delta_interval_lower = None
        self.f_delta_interval_upper = None

    @abstractmethod
    def get_bootstrap_data(self, instances: FeatureData) -> FeatureData:
        """Get the data points to use for interval estimation.

        param instances: FeatureData: The feature data to fit the bootstrap system on.
        return FeatureData: The feature data to use for interval estimation.
        """
        raise NotImplementedError

    def transform(self, instances: FeatureData) -> LLRData:
        """Transform the provided instances to include the best estimate and confidence intervals.

        param instances: FeatureData: The feature data to transform.
        return LLRData: The transformed feature data with best estimate and confidence intervals.
        """
        if self.f_delta_interval_lower is None or self.f_delta_interval_upper is None:
            raise ValueError('Bootstrap intervals have not been computed. Please fit the bootstrap first.')

        best_estimate = super().transform(instances)
        best_1d_estimate = best_estimate.features.reshape(-1)
        interval_lower = best_1d_estimate + self.f_delta_interval_lower(best_1d_estimate)
        interval_upper = best_1d_estimate + self.f_delta_interval_upper(best_1d_estimate)

        return best_estimate.replace_as(
            LLRData, features=np.stack([best_1d_estimate, interval_lower, interval_upper], axis=1)
        )

    def fit_transform(self, instances: FeatureData) -> LLRData:
        """Combine fitting and transforming in one step.

        param instances: FeatureData: The feature data to fit and transform.
        return LLRData: The transformed feature data with best estimate and confidence intervals.
        """
        return self.fit(instances).transform(instances)

    def fit(self, instances: FeatureData) -> Self:
        """Fit the bootstrap system to the provided instances.

        param instances: FeatureData: The feature data to fit the bootstrap system on.
        return Self: The fitted bootstrap system.
        """

        all_vals = []
        rng = np.random.default_rng(self.seed)

        bootstrap_data = self.get_bootstrap_data(instances)

        for _ in range(self.n_bootstraps):
            sample_index = rng.choice(len(instances), size=len(instances))
            samples = instances[sample_index]
            super().fit(samples)
            all_vals.append(super().transform(bootstrap_data).features.reshape(-1))

        all_vals = np.stack(all_vals, axis=1)

        intervals = np.quantile(all_vals, self.interval, axis=1)

        super().fit(instances)
        best_estimate_bootstrap_data = super().transform(bootstrap_data)
        x_vals = best_estimate_bootstrap_data.features.reshape(-1)

        lower = intervals[0] - x_vals
        upper = intervals[1] - x_vals

        # Use the numeric 1D feature array as x for interpolation (not the FeatureData object).
        self.f_delta_interval_lower = interp1d(x_vals, lower, bounds_error=False, fill_value=(lower[0], lower[-1]))
        self.f_delta_interval_upper = interp1d(x_vals, upper, bounds_error=False, fill_value=(upper[0], upper[-1]))
        return self


class BootstrapAtData(Bootstrap):
    """Bootstrap system that uses the original training data points for interval estimation.
    See the Bootstrap class for more details.
    """

    def get_bootstrap_data(self, instances: FeatureData) -> FeatureData:
        """Get the data points to use for interval estimation. The original training data points are used.

        param instances: FeatureData: The feature data to fit the bootstrap system on.
        return FeatureData: The feature data to use for interval estimation.
        """
        return instances


class BootstrapEquidistant(Bootstrap):
    """Bootstrap system that uses equidistant points within the range of the training data for interval estimation.
    See the Bootstrap class for more details.
    """

    def __init__(
        self,
        steps: list[tuple[str, Any]],
        n_bootstraps: int = 400,
        interval: tuple[float, float] = (0.05, 0.95),
        seed: int | None = None,
        n_points: int | None = 1000,
    ):
        """Initialize the instance with the given pipeline steps, number of bootstraps, and interval.

        Parameters:
        param steps: list[tuple[str, Any]]: The steps of the pipeline to be bootstrapped.
        param n_bootstraps: int: The number of bootstrap samples to generate. Default is 400.
        param interval: tuple[float, float]: The lower and upper quantiles for the confidence interval.
                                             Default: (0.05,0.95).
        param n_points: int | None: The number of equidistant points to use for interval estimation. Default is 1000.
                                    If None, uses the number of instances in the training data.
        """
        super().__init__(steps, n_bootstraps, interval, seed)
        self.n_points = n_points

    def get_bootstrap_data(self, instances: FeatureData) -> FeatureData:
        """Get the data points to use for interval estimation. This is done by creating equidistant points
        within the range of the training data.

        param instances: FeatureData: The feature data to fit the bootstrap system on.
        return FeatureData: The feature data to use for interval estimation.
        """
        if instances.features.ndim != 2 or instances.features.shape[1] != 1:
            raise ValueError(f'expected 2D feature array with 1 column; found shape: {instances.features.shape}')

        feat_vals = instances.features.reshape(-1)

        if self.n_points is None:
            self.n_points = len(instances)

        equidistant_indices = np.linspace(
            start=np.min(feat_vals), stop=np.max(feat_vals), num=self.n_points, dtype=instances.features.dtype
        ).reshape(-1, 1)
        return FeatureData(features=equidistant_indices)


@config_parser
def bootstrap(modules_config: ContextAwareDict, output_dir: Path) -> BootstrapAtData:
    """
    Transitional function to parse a bootstrapping pipeline.

    The configuration takes the following arguments:
    - steps: the pipeline steps to bootstrap
    - points: the points to bootstrap, can be either `data` or `equidistant`. This selects the bootstrapping class:
      `BootstrapAtData` for 'data', or `BootstrapEquidistant` for 'equidistant'

    Any other arguments are passed directly to the `__init__()` method of the bootstrapping class.

    :param modules_config: the configuration
    :param output_dir: where to write output, if any
    :return: a bootstrapping object
    """
    bootstrap_methods = {
        'data': BootstrapAtData,
        'equidistant': BootstrapEquidistant,
    }

    bootstrap_method = pop_field(modules_config, 'points', validate=bootstrap_methods.get)
    steps = parse_steps(pop_field(modules_config, 'steps', validate=check_not_none), output_dir)
    return bootstrap_method(steps, **modules_config)
