from collections.abc import Iterator

import numpy as np
import sklearn
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold

from lir.data.models import DataSet, DataStrategy


class BinaryTrainTestSplit(DataStrategy):
    """Representation of a regular, binary train/test split fold.

    The BinaryTrainTestSplit implementation is aimed at data consisting of two classes.
    """

    def __init__(self, source: DataSet, test_size: float | int, seed: int | None = None):
        self.source = source
        self.test_size = test_size
        self.seed = seed

    def __iter__(self) -> Iterator:
        """Allow iteration by looping over the resulting train/test split(s)."""
        instances = self.source.get_instances()

        indexes = np.arange(len(instances))
        indexes_train, indexes_test = sklearn.model_selection.train_test_split(
            indexes, stratify=instances.labels, test_size=self.test_size, shuffle=True, random_state=self.seed
        )

        yield instances[indexes_train], instances[indexes_test]


class BinaryCrossValidation(DataStrategy):
    """Representation of a K-fold cross validation iterator over each train/test split fold.

    The BinaryCrossValidation implementation is aimed at data consisting of two classes.
    """

    def __init__(self, source: DataSet, folds: int, seed: int | None = None):
        self.source = source
        self.folds = folds
        self.seed = seed
        self.shuffle = True if self.seed is not None else False  # noqa: SIM210

    def __iter__(self) -> Iterator:
        """Allow iteration by looping over the resulting train/test split(s)."""
        kf = KFold(n_splits=self.folds, shuffle=self.shuffle, random_state=self.seed)
        instances = self.source.get_instances()
        for _i, (train_index, test_index) in enumerate(kf.split(instances.features, y=instances.labels)):
            yield instances[train_index], instances[test_index]


class MulticlassTrainTestSplit(DataStrategy):
    """Representation of a multi-class train/test split.

    The MulticlassTrainTestSplit implementation is aimed at data consisting of multiple classes.
    """

    def __init__(self, source: DataSet, test_size: float | int, seed: int | None = None):
        self.source = source
        self.test_size = test_size
        self.seed = seed

    def __iter__(self) -> Iterator:
        """Allow iteration by looping over the resulting train/test split(s)."""
        splitter = GroupShuffleSplit(test_size=self.test_size, n_splits=1, random_state=self.seed)
        instances = self.source.get_instances()
        ((train_index, test_index),) = splitter.split(instances.features, instances.labels, instances.labels)

        yield instances[train_index], instances[test_index]


class MulticlassCrossValidation(DataStrategy):
    """Representation of a K-fold cross validation iterator over each train/test split fold.

    The MulticlassCrossValidation implementation is aimed at classification data consisting of multiple classes.
    """

    def __init__(self, source: DataSet, folds: int):
        self.source = source
        self.folds = folds

    def __iter__(self) -> Iterator:
        """Allow iteration by looping over the resulting train/test split(s)."""
        kf = GroupKFold(n_splits=self.folds)

        instances = self.source.get_instances()
        for _i, (train_index, test_index) in enumerate(kf.split(instances.features, groups=instances.labels)):
            yield instances[train_index], instances[test_index]
