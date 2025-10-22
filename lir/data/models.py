from abc import abstractmethod, ABC
from typing import Tuple, Iterable

import numpy as np


class DataSet(ABC):
    """General representation of a data source.

    Each data source should provide access to a feature vector, a target (ground truth labels) vector, and a meta data
    vector, by implementing the `get_instances()` method.
    """

    @abstractmethod
    def get_instances(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns a tuple of instances and their meta data. For a data set of `n`
        instances, this function returns a tuple of:

        - an array of instances with at least one dimension of size `n`;
        - an array of ground truth labels with dimensions `(n,)`;
        - an array of meta data with at least one dimension of size `n`.
        """

        raise NotImplementedError


class DataStrategy(ABC, Iterable):
    """
    General representation of a data setup strategy.

    All subclasses must implement a `__iter__()` method.

    The custom __iter__() method should return an iterator over tuples
    of a training set and a test set. Both the training set and the test
    set consist of a tuple of instances, labels, and meta data.
    """

    pass
