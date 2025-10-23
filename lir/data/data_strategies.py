from typing import Iterator

import sklearn
from sklearn.model_selection import KFold, GroupKFold, GroupShuffleSplit

from lir.data.models import DataSet, DataStrategy


class BinaryTrainTestSplit(DataStrategy):
    """Representation of a regular, binary train/test split fold.

    The BinaryTrainTestSplit implementation is aimed at data consisting of two classes.
    """

    def __init__(self, source: DataSet, test_size: float | int, seed: int | None = None):
        self.source = source
        self.test_size = test_size
        self.seed = seed
        self.shuffle = True if self.seed is not None else False

    def __iter__(self) -> Iterator:
        """Allow iteration by looping over the resulting train/test split(s)."""
        (
            features_train,
            features_test,
            labels_train,
            labels_test,
            meta_train,
            meta_test,
        ) = sklearn.model_selection.train_test_split(
            *self.source.get_instances(),
            test_size=self.test_size,
            shuffle=self.shuffle,
            random_state=self.seed,
        )
        yield (
            (features_train, labels_train, meta_train),
            (
                features_test,
                labels_test,
                meta_test,
            ),
        )


class BinaryCrossValidation(DataStrategy):
    """Representation of a K-fold cross validation iterator over each train/test split fold.

    The BinaryCrossValidation implementation is aimed at data consisting of two classes.
    """

    def __init__(self, source: DataSet, folds: int, seed: int | None = None):
        self.source = source
        self.folds = folds
        self.seed = seed
        self.shuffle = True if self.seed is not None else False

    def __iter__(self) -> Iterator:
        """Allow iteration by looping over the resulting train/test split(s)."""
        kf = KFold(n_splits=self.folds, shuffle=self.shuffle, random_state=self.seed)
        features, labels, meta = self.source.get_instances()
        for i, (train_index, test_index) in enumerate(kf.split(features, y=labels)):
            yield (
                (features[train_index], labels[train_index], meta[train_index]),
                (
                    features[test_index],
                    labels[test_index],
                    meta[test_index],
                ),
            )


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
        features, labels, meta = self.source.get_instances()
        ((train_index, test_index),) = splitter.split(features, labels, labels)

        training_set = features[train_index], labels[train_index], meta[train_index]
        test_set = features[test_index], labels[test_index], meta[test_index]
        yield training_set, test_set


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

        features, labels, meta = self.source.get_instances()
        for i, (train_index, test_index) in enumerate(kf.split(features, groups=labels)):
            yield (
                (features[train_index], labels[train_index], meta[train_index]),
                (
                    features[test_index],
                    labels[test_index],
                    meta[test_index],
                ),
            )
