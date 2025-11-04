from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import TransformerMixin


class LLRBounder(TransformerMixin, ABC):
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
    def _validate(llrs: np.ndarray, labels: np.ndarray) -> None:
        if len(llrs.shape) != 1:
            raise ValueError(f'llrs argument should be 1-dimensional; dimensions found: {len(llrs.shape)}')
        if len(labels.shape) != 1:
            raise ValueError(f'labels argument should be 1-dimensional; dimensions found: {len(labels.shape)}')
        if llrs.shape[0] != labels.shape[0]:
            raise ValueError(
                f'number of labels does not match the number of llrs ({labels.shape[0]} != {llrs.shape[0]})'
            )
        if list(np.unique(labels)) != [0, 1]:
            raise ValueError(f'labels expected: 0, 1; found: {np.unique(labels)}')

    def fit(self, llrs: np.ndarray, labels: np.ndarray) -> 'LLRBounder':
        """
        Configures this bounder by calculating bounds.

        assuming that y=1 corresponds to Hp, y=0 to Hd
        """
        # validate the input
        self._validate(llrs, labels)

        # calculate the bounds
        self.lower_llr_bound, self.upper_llr_bound = self.calculate_bounds(llrs, labels)

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

    def transform(self, llrs: np.ndarray) -> np.ndarray:
        """
        a transform entails calling the first step calibrator and applying the bounds found
        """
        if self.lower_llr_bound is not None:
            llrs = np.where(self.lower_llr_bound < llrs, llrs, self.lower_llr_bound)
        if self.upper_llr_bound is not None:
            llrs = np.where(self.upper_llr_bound > llrs, llrs, self.upper_llr_bound)
        return llrs


class StaticBounder(LLRBounder):
    """
    Bound LLRs to constant values.

    This bounder takes arguments for a lower and upper bound, which may take `None` in which case no bounds are applied.
    """

    def __init__(self, lower_llr_bound: float, upper_llr_bound: float):
        super().__init__(lower_llr_bound, upper_llr_bound)

    def calculate_bounds(self, llrs: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
        return self.lower_llr_bound, self.upper_llr_bound
