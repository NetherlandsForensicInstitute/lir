from collections.abc import Iterator
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from lir import DataStrategy, InstanceData
from lir.util import check_type


def is_valid_input(instances: InstanceData) -> bool:  # numpydoc ignore=PR01,RT01
    """Return True iff pair-based strategies can be applied."""
    return (
        instances.source_ids is not None and len(instances.source_ids.shape) == 2 and instances.source_ids.shape[1] != 2
    )


def _check_input(instances: InstanceData) -> None:  # numpydoc ignore=PR01
    """Raise an error unless pair-based strategies can be applied."""
    if instances.source_ids is None or len(instances.source_ids.shape) != 2 or instances.source_ids.shape[1] != 2:
        raise ValueError(f'expected two-column source_ids; shape found: {getattr(instances.source_ids, "shape", None)}')


class PairsTrainTestSplit(DataStrategy):
    """
    A train/test split policy for paired instances.

    The input data should have ``source_ids`` with two columns. This split assigns all sources to either the training
    set or the test set. The pairs are assigned to training or testing if both of their sources have that role. Pairs
    with mixed roles are omitted.

    Parameters
    ----------
    test_size : float | int
        Fraction or absolute number of items assigned to the test split.
    seed : int | None
        Random seed controlling stochastic behaviour for reproducible results.
    """

    def __init__(self, test_size: float | int, seed: int | None = None):
        self.test_size = test_size
        self.seed = seed

    def apply[DataType: InstanceData](self, instances: DataType) -> Iterator[tuple[DataType, DataType]]:
        """
        Split the data into a training set and a test set.

        Parameters
        ----------
        instances : InstanceDataType
            Input instances to be processed by this method.

        Yields
        ------
        tuple[DataType, DataType]
            An iterator over a single item, which is a tuple of the training set and the test set.
        """
        _check_input(instances)
        source_ids = instances.source_ids
        sources_train, sources_test = train_test_split(
            np.unique(check_type(np.ndarray, source_ids)),
            test_size=self.test_size,
            shuffle=True,
            random_state=self.seed,
        )

        sources_train = set(sources_train)

        def use_in_training(value: Any) -> bool:
            return value in sources_train

        training_source_ids = np.vectorize(use_in_training)(instances.source_ids)
        training_instances = np.all(training_source_ids, axis=1)
        test_instances = np.all(~training_source_ids, axis=1)

        yield instances[training_instances], instances[test_instances]
