from collections.abc import Iterator

import numpy as np
from sklearn.model_selection import KFold, train_test_split

from lir import DataStrategy, InstanceData


def is_valid_input(instances: InstanceData) -> bool:  # numpydoc ignore=PR01,RT01
    """Return True iff label-based strategies can be applied."""
    return instances.hypothesis is not None


def _check_input(instances: InstanceData) -> None:  # numpydoc ignore=PR01
    """Raise an error unless label-based strategies can be applied."""
    if instances.hypothesis is None:
        raise ValueError('unable to perform train/test split by hypothesis labels without labels')


class TrainTestSplit(DataStrategy):
    """
    Split the data into a training set and a test set.

    This splitter distributes the instances randomly over a training set and test set. Each instance is assigned to
    either the training set or the test set, but no sources will have instances that appear in both. The hypothesis
    labels are used to distribute the instances of each hypothesis proportionally to both sets.

    This splitter is suitable for most specific-source setups. If you have a common-source setup, take a look at
    ``SourcesTrainTestSplit``. Alternatively, use the ``CrossValidation`` strategy for cross-validation.

    The input data should have hypothesis labels. This split assigns instances of both classes to the training set and
    the test set.

    In an experiment setup file, the split strategy can be referenced as:

    .. code-block:: yaml

        splits:
          strategy: train_test
          test_size: 0.2  # the (hold-out) test set  is 20% of the data
          seed: 42  # optional

    Parameters
    ----------
    test_size : float | int
        Size of the test set. If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to
        include in the test split. If int, represents the absolute number of test samples.
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
        instances : DataType
            Instances to split.

        Yields
        ------
        tuple[DataType, DataType]
            An iterator over a single item, which is a tuple of the training set and the test set.
        """
        indexes = np.arange(len(instances))
        indexes_train, indexes_test = train_test_split(
            indexes, stratify=instances.require_labels, test_size=self.test_size, shuffle=True, random_state=self.seed
        )

        yield instances[indexes_train], instances[indexes_test]


class CrossValidation(DataStrategy):
    """
    K-fold cross-validation iterator over successive train/test splits.

    The input data must contain **hypothesis labels**. If the data has ``source_ids`` but not hypothesis labels, use
    :class:`~lir.data_strategies.SourcesCrossValidation` instead.

    Each fold is constructed so that instances from both hypotheses are present in every split.

    This strategy may be registered in a YAML registry as follows:

    .. code-block:: yaml

        splits:
          strategy: cross_validation
          folds: 5  # the number k in k-fold cross-validation
          seed: 42  # optional

    Parameters
    ----------
    folds : int
        Number of cross-validation folds to generate.
    shuffle : bool | None
        Whether to shuffle the data splitting. If `None`, the data will be shuffled if `random_state` is not `None`.
    seed : int | None
        Random seed controlling stochastic behaviour for reproducible results.
    """

    def __init__(self, folds: int, shuffle: bool | None = None, seed: int | None = None):
        self.folds = folds
        self.seed = seed
        self.shuffle: bool = shuffle or (shuffle is None and seed is not None)

    def apply[DataType: InstanceData](self, instances: DataType) -> Iterator[tuple[DataType, DataType]]:
        """
        Return an iterator over *k* train/test splits.

        Parameters
        ----------
        instances : InstanceDataType
            Input instances to be processed by this method.
        """
        kf = KFold(n_splits=self.folds, shuffle=self.shuffle, random_state=self.seed)
        for train_index, test_index in kf.split(np.arange(len(instances)), y=instances.hypothesis):
            yield instances[train_index], instances[test_index]
