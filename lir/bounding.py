import logging
from abc import ABC, abstractmethod
from typing import Self

import numpy as np

from lir import Transformer
from lir.data.models import FeatureData, LLRData


LOG = logging.getLogger(__name__)


class LLRBounder(Transformer, ABC):
    """
    Base class for LLR bounders.

    A bounder updates any LLRs that are out of bounds. Any LLR values within bounds remain unchanged. LLR values that
    are out-of-bounds are updated tot the nearest bound.
    """

    def __init__(
        self,
        lower_llr_bound: float | None = None,
        upper_llr_bound: float | None = None,
    ):
        self.lower_llr_bound = lower_llr_bound
        self.upper_llr_bound = upper_llr_bound

    @abstractmethod
    def calculate_bounds(self, llrs: np.ndarray, labels: np.ndarray) -> tuple[float | None, float | None]:
        """
        Calculates and returns appropriate bounds for a set of LLRs and their labels.
        """
        raise NotImplementedError

    @staticmethod
    def _validate(instances: FeatureData) -> LLRData:
        if not isinstance(instances, LLRData):
            LOG.info(f'casting `{type(instances)}` to `LLRData`')
            instances = instances.replace_as(LLRData)
        return instances

    def fit(self, instances: FeatureData) -> Self:
        """
        Configures this bounder by calculating bounds.

        assuming that y=1 corresponds to Hp, y=0 to Hd
        """
        instances = self._validate(instances)

        if instances.labels is None:
            raise ValueError(f'{type(self)}.fit() requires labeled data')

        # calculate the bounds
        self.lower_llr_bound, self.upper_llr_bound = self.calculate_bounds(instances.llrs, instances.labels)

        # check the sanity of the bounds
        if (
            self.lower_llr_bound is not None
            and self.upper_llr_bound is not None
            and self.lower_llr_bound > self.upper_llr_bound
        ):
            raise ValueError(
                'the lower bound must be lower than the upper bound; '
                f'lower_llr_bound={self.lower_llr_bound}; upper_llr_bound={self.upper_llr_bound}'
            )

        return self

    def transform(self, instances: FeatureData) -> LLRData:
        """
        a transform entails calling the first step calibrator and applying the bounds found
        """
        instances = self._validate(instances)

        llrs = instances.features
        if self.lower_llr_bound is not None:
            llrs = np.where(self.lower_llr_bound < llrs, llrs, self.lower_llr_bound)
        if self.upper_llr_bound is not None:
            llrs = np.where(self.upper_llr_bound > llrs, llrs, self.upper_llr_bound)
        return instances.replace(features=llrs)


class StaticBounder(LLRBounder):
    """
    Bound LLRs to constant values.

    This bounder takes arguments for a lower and upper bound, which may take `None` in which case no bounds are applied.
    """

    def __init__(self, lower_llr_bound: float, upper_llr_bound: float):
        super().__init__(lower_llr_bound, upper_llr_bound)

    def calculate_bounds(self, llrs: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
        return self.lower_llr_bound, self.upper_llr_bound
