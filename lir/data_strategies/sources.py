from collections.abc import Iterator

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from lir import DataStrategy, InstanceData


class SourcesTrainTestSplit(DataStrategy):
    """
    Split the data into a training set and a test set by their source ids.

    This splitter uses the ``source_ids`` attribute and distributes the sources over the training and test set. Each
    source is assigned to either the training set or the test set, but no sources will have instances that appear in
    both.

    This splitter is suitable for most common-source setups. Alternatively, use the ``SourcesCrossValidation`` strategy
    for cross-validation.

    In an experiment setup file, the split strategy can be referenced as:

    .. code-block:: yaml

        splits:
          strategy: train_test_sources
          test_size: 0.5  # the proportion of sources in the test set
          seed: 42        # optional

    This class internally uses ``sklearn.model_selection.GroupShuffleSplit``.

    Parameters
    ----------
    test_size : float | int
        Fraction or absolute number of items assigned to the test split. If float, should be between 0.0 and 1.0 and
        represent the proportion of sources to include inthe test split (rounded up). If int, represents the absolute
        number of test sources.
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
        splitter = GroupShuffleSplit(test_size=self.test_size, n_splits=1, random_state=self.seed)
        ((train_index, test_index),) = splitter.split(np.arange(len(instances)), groups=instances.source_ids_1d)

        yield instances[train_index], instances[test_index]


class SourcesCrossValidation(DataStrategy):
    """
    K-fold cross-validation by source id.

    This data strategy uses the ``source_ids`` attribute and distributes the sources over *k* different subsets. If the
    data have hypothesis labels, use :class:`~lir.data_strategies.CrossValidation` instead.

    Each of the subsets will be offered once as the test set, using the others as the training set. Each source is
    assigned to exactly one of the subsets, and no sources will have instances that appear in more than one.

    In an experiment setup file, the data strategy can be referenced as:

    .. code-block:: yaml

        splits:
          strategy: cross_validation_sources
          folds: 5
          random_state: 0

    This class internally uses :class:`~sklearn.model_selection.GroupKFold`.

    Parameters
    ----------
    folds : int
        Number of cross-validation folds to generate.
    shuffle : bool | None
        Whether to shuffle the groups before splitting into batches. Note that the samples within each split will not be
        shuffled. If `None`, the data will be shuffled if `random_state` is not `None`.
    random_state : int | None
        When shuffle is True, random_state affects the ordering of the indices, which controls the randomness of each
        fold. Otherwise, this parameter has no effect. Pass an int for reproducible output across multiple function
        calls.
    """

    def __init__(self, folds: int, shuffle: bool | None = None, random_state: int | None = None):
        if shuffle is None:
            shuffle = random_state is not None
        random_state = random_state
        self._kf = GroupKFold(n_splits=folds, shuffle=shuffle, random_state=random_state)

    def apply[DataType: InstanceData](self, instances: DataType) -> Iterator[tuple[DataType, DataType]]:
        """
        Perform *k*-fold cross-validation.

        Return an iterator over *k* train/test splits.

        Parameters
        ----------
        instances : InstanceDataType
            Input instances to be processed by this method.
        """
        for train_index, test_index in self._kf.split(np.arange(len(instances)), groups=instances.source_ids_1d):
            yield instances[train_index], instances[test_index]
