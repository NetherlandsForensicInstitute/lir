from typing import Any, Self

import numpy as np
from scipy.interpolate import interp1d

from lir.data.models import FeatureData, LLRData
from lir.transform.pipeline import Pipeline


class BootstrapAtData(Pipeline):
    """Bootstrap system that estimates confidence intervals around the best estimate of a pipeline.
    This bootstrap system creates bootstrap samples from the training data, fits the pipeline on each sample,
    and then computes confidence intervals for the pipeline outputs based on the variability across the bootstrap
    samples.

    Attributes:
        interval: The lower and upper quantiles for the confidence interval.
        n_bootstraps: The number of bootstrap samples to generate.
        f_interval_lower: Interpolation function for the lower bound of the interval.
        f_interval_upper: Interpolation function for the upper bound of the interval.

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
        """
        self.interval = interval
        self.n_bootstraps = n_bootstraps
        self.seed = seed

        self.f_interval_lower = None
        self.f_interval_upper = None

        super().__init__(steps)

    def fit(self, instances: FeatureData) -> Self:
        """Fit the bootstrap system to the provided instances.

        param instances: FeatureData: The feature data to fit the bootstrap system on.
        return Self: The fitted bootstrap system.
        """

        all_vals = []
        rng = np.random.default_rng(self.seed)

        for _ in range(self.n_bootstraps):
            sample_index = rng.choice(len(instances), size=len(instances))
            samples = instances[sample_index]
            super().fit(samples)
            all_vals.append(super().transform(instances).features.reshape(-1))

        all_vals = np.stack(all_vals, axis=1)

        intervals = np.quantile(all_vals, self.interval, axis=1)
        best_estimate = super().fit_transform(instances)

        best_estimate_values = best_estimate.features.reshape(-1)
        self.f_interval_lower = interp1d(best_estimate_values, intervals[0] - best_estimate_values)
        self.f_interval_upper = interp1d(best_estimate_values, intervals[1] - best_estimate_values)
        return self

    def transform(self, instances: FeatureData) -> LLRData:
        """Transform the provided instances to include the best estimate and confidence intervals.

        param instances: FeatureData: The feature data to transform.
        return LLRData: The transformed feature data with best estimate and confidence intervals.
        """
        if self.f_interval_lower is None or self.f_interval_upper is None:
            raise ValueError('Bootstrap intervals have not been computed. Please fit the bootstrap first.')

        best_estimate = super().transform(instances)
        best_1d_estimate = best_estimate.features.reshape(-1)
        interval_lower = self.f_interval_lower(best_1d_estimate)
        interval_upper = self.f_interval_upper(best_1d_estimate)

        return best_estimate.replace_as(
            LLRData, features=np.stack([best_1d_estimate, interval_lower, interval_upper], axis=1)
        )

    def fit_transform(self, instances: FeatureData) -> LLRData:
        """Combine fitting and transforming in one step.

        param instances: FeatureData: The feature data to fit and transform.
        return LLRData: The transformed feature data with best estimate and confidence intervals.
        """
        return self.fit(instances).transform(instances)


class BootstrapEquidistant(Pipeline):
    """Bootstrap system that estimates confidence intervals around the best estimate of a pipeline.
    This bootstrap system creates bootstrap samples from the training data, fits the pipeline on each sample,
    and then computes confidence intervals for the pipeline outputs based on the variability across the bootstrap
    samples.

    Attributes:
        interval: The lower and upper quantiles for the confidence interval.
        n_bootstraps: The number of bootstrap samples to generate.
        f_interval_lower: Interpolation function for the lower bound of the interval.
        f_interval_upper: Interpolation function for the upper bound of the interval.

    """

    def __init__(
        self,
        steps: list[tuple[str, Any]],
        n_bootstraps: int = 400,
        interval: tuple[float, float] = (0.05, 0.95),
        seed: int | None = None,
        n_points: int | None = 1000,
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
        self.interval = interval
        self.n_bootstraps = n_bootstraps
        self.seed = seed
        self.n_points = n_points
        self.f_interval_lower = None
        self.f_interval_upper = None

        super().__init__(steps)

    def fit(self, instances: FeatureData) -> Self:
        """Fit the bootstrap system to the provided instances.

        param instances: FeatureData: The feature data to fit the bootstrap system on.
        return Self: The fitted bootstrap system.
        """

        all_vals = []
        rng = np.random.default_rng(self.seed)

        # Raise an exception if the dimensions are not what we expect.
        if instances.features.ndim != 2:
            raise ValueError('Expected 2D feature array.')

        feat_vals = instances.features.reshape(-1)

        if self.n_points is None:
            self.n_points = len(instances)
        equidistant_indices = np.linspace(
            np.min(feat_vals), np.max(feat_vals), num=self.n_points, dtype=instances.features.dtype
        ).reshape(-1, 1)

        # Create FeatureData so the points can be used in the pipeline.
        fd = FeatureData(features=equidistant_indices)

        for _ in range(self.n_bootstraps):
            sample_index = rng.choice(len(instances), size=len(instances))
            samples = instances[sample_index]
            super().fit(samples)
            all_vals.append(super().transform(fd).features.reshape(-1))

        all_vals = np.stack(all_vals, axis=1)

        intervals = np.quantile(all_vals, self.interval, axis=1)

        super().fit(instances)
        best_on_equidistant = super().transform(fd).features.reshape(-1)

        lower = intervals[0] - best_on_equidistant
        upper = intervals[1] - best_on_equidistant

        self.f_interval_lower = interp1d(
            best_on_equidistant, lower, bounds_error=False, fill_value=(lower[0], lower[-1])
        )
        self.f_interval_upper = interp1d(
            best_on_equidistant, upper, bounds_error=False, fill_value=(upper[0], upper[-1])
        )

        return self

    def transform(self, instances: FeatureData) -> LLRData:
        """Transform the provided instances to include the best estimate and confidence intervals.

        param instances: FeatureData: The feature data to transform.
        return LLRData: The transformed feature data with best estimate and confidence intervals.
        """
        if self.f_interval_lower is None or self.f_interval_upper is None:
            raise ValueError('Bootstrap intervals have not been computed. Please fit the bootstrap first.')

        best_estimate = super().transform(instances)
        best_1d_estimate = best_estimate.features.reshape(-1)
        interval_lower = self.f_interval_lower(best_1d_estimate)
        interval_upper = self.f_interval_upper(best_1d_estimate)

        return best_estimate.replace_as(
            LLRData, features=np.stack([best_1d_estimate, interval_lower, interval_upper], axis=1)
        )

    def fit_transform(self, instances: FeatureData) -> LLRData:
        """Combine fitting and transforming in one step.

        param instances: FeatureData: The feature data to fit and transform.
        return LLRData: The transformed feature data with best estimate and confidence intervals.
        """
        return self.fit(instances).transform(instances)
