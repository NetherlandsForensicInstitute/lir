import logging
from abc import ABC, abstractmethod
from typing import Self

import numpy as np

from lir import Transformer
from lir.data.models import FeatureData, InstanceData, LLRData
from lir.util import check_type


LOG = logging.getLogger(__name__)


class LLRBounder(Transformer, ABC):
    """
    Base class for LLR bounders.

    A bounder updates any LLRs that are out of bounds. Any LLR values within bounds remain unchanged. LLR values that
    are out-of-bounds are updated to the nearest bound.
    """

    def __init__(
        self,
        lower_llr_bound: float | None = None,
        upper_llr_bound: float | None = None,
    ):
        self.lower_llr_bound = lower_llr_bound
        self.upper_llr_bound = upper_llr_bound

    @abstractmethod
    def calculate_bounds(self, llrdata: LLRData) -> tuple[float | None, float | None]:
        """
        Calculates and returns appropriate bounds for a set of LLRs and their labels.
        """
        raise NotImplementedError

    @staticmethod
    def _validate(instances: InstanceData) -> LLRData:
        instances = check_type(FeatureData, instances)
        if not isinstance(instances, LLRData):
            LOG.info(f'casting `{type(instances)}` to `LLRData`')
            instances = instances.replace_as(LLRData)
        return instances

    def fit(self, instances: InstanceData) -> Self:
        """
        Configures this bounder by calculating bounds.

        assuming that y=1 corresponds to Hp, y=0 to Hd
        """
        instances = self._validate(instances)

        if instances.labels is None:
            raise ValueError(f'{type(self)}.fit() requires labeled data')

        # calculate the bounds
        self.lower_llr_bound, self.upper_llr_bound = self.calculate_bounds(instances)

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

    def apply(self, instances: InstanceData) -> LLRData:
        """
        a transform entails calling the first step calibrator and applying the bounds found
        """
        instances = self._validate(instances)

        llrs = instances.features

        # Clip the LLRs to the bounds, where np.clip handles the None values correctly.
        llrs = np.clip(llrs, self.lower_llr_bound, self.upper_llr_bound)

        return instances.replace(
            features=llrs, llr_upper_bound=self.upper_llr_bound, llr_lower_bound=self.lower_llr_bound
        )


class StaticBounder(LLRBounder):
    """
    Bound LLRs to constant values.

    This bounder takes arguments for a lower and upper bound, which may take `None` in which case no bounds are applied.
    """

    def __init__(self, lower_llr_bound: float | None, upper_llr_bound: float | None):
        super().__init__(lower_llr_bound, upper_llr_bound)

    def calculate_bounds(self, llrdata: LLRData) -> tuple[float | None, float | None]:
        return self.lower_llr_bound, self.upper_llr_bound


class NSourceBounder(LLRBounder):
    """
    Bound LLRs based on the number of sources.

    This bounder sets the lower LLR bound to -log(N) and the upper bound to log(N), where N is the number of sources.

    In non-log space, this corresponds to bounding likelihood ratios to [1/N, N]. This is a logical consequence of
    having N sources: no source can provide more than N support for one hypothesis over the other.
    """

    def calculate_bounds(self, llrdata: LLRData) -> tuple[float | None, float | None]:
        if llrdata.source_ids is None:
            raise ValueError(f'{type(self)} requires source IDs to calculate bounds')

        n_sources = np.unique(llrdata.source_ids, sorted=False)
        log_n_sopurces = np.log10(len(n_sources))
        return -log_n_sopurces, log_n_sopurces
