from collections.abc import Iterable, Iterator
from enum import Enum
from typing import Any

import numpy as np
import sklearn
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold

from lir.config.base import check_not_none
from lir.data.models import DataStrategy, FeatureData


class BinaryTrainTestSplit(DataStrategy):
    """Representation of a train/test split.

    The input data should have hypothesis labels. This split assigns instances of both classes to the training set and
    the test set.
    """

    def __init__(self, test_size: float | int, seed: int | None = None):
        self.test_size = test_size
        self.seed = seed

    def apply(self, instances: FeatureData) -> Iterable[tuple[FeatureData, FeatureData]]:
        """Allow iteration by looping over the resulting train/test split(s)."""
        indexes = np.arange(len(instances))
        indexes_train, indexes_test = sklearn.model_selection.train_test_split(
            indexes, stratify=instances.labels, test_size=self.test_size, shuffle=True, random_state=self.seed
        )

        yield instances[indexes_train], instances[indexes_test]


class BinaryCrossValidation(DataStrategy):
    """
    K-fold cross-validation iterator over successive train/test splits.

    The input data must contain class labels. Each fold is constructed so that
    instances from both classes are present in every split.

    This strategy may be registered in a YAML registry as follows:

    .. code-block:: yaml

        data:
          [...]
          splits:
            strategy: binary_cross_validation
            folds: 5
            seed: 42

    """

    def __init__(self, folds: int, seed: int | None = None):
        self.folds = folds
        self.seed = seed
        self.shuffle = True if self.seed is not None else False  # noqa: SIM210

    def apply(self, instances: FeatureData) -> Iterable[tuple[FeatureData, FeatureData]]:
        """Allow iteration by looping over the resulting train/test split(s)."""
        kf = KFold(n_splits=self.folds, shuffle=self.shuffle, random_state=self.seed)
        for _i, (train_index, test_index) in enumerate(kf.split(instances.features, y=instances.labels)):
            yield instances[train_index], instances[test_index]


class MulticlassTrainTestSplit(DataStrategy):
    """Representation of a multi-class train/test split.

    The input data should have source_ids. This split assigns all instances of a source to either the training set or
    the test set.
    """

    def __init__(self, test_size: float | int, seed: int | None = None):
        self.test_size = test_size
        self.seed = seed

    def apply(self, instances: FeatureData) -> Iterable[tuple[FeatureData, FeatureData]]:
        """Allow iteration by looping over the resulting train/test split(s)."""
        splitter = GroupShuffleSplit(test_size=self.test_size, n_splits=1, random_state=self.seed)
        ((train_index, test_index),) = splitter.split(instances.features, instances.source_ids, instances.source_ids)

        yield instances[train_index], instances[test_index]


class MulticlassCrossValidation(DataStrategy):
    """
    K-fold cross-validation iterator over successive train/test splits.

    The input data must contain ``source_ids``. All instances originating from the
    same source are assigned to the same fold, ensuring that no source appears in
    both the training and test sets within a split.

    In a benchmark configuration, the split strategy can be referenced as:

    .. code-block:: yaml

        data:
          [...]
          splits:
            strategy: multiclass_cross_validation
            folds: 5
    """

    def __init__(self, folds: int):
        self.folds = folds

    def apply(self, instances: FeatureData) -> Iterable[tuple[FeatureData, FeatureData]]:
        """Allow iteration by looping over the resulting train/test split(s)."""
        kf = GroupKFold(n_splits=self.folds)

        for _i, (train_index, test_index) in enumerate(kf.split(instances.features, groups=instances.source_ids_1d)):
            yield instances[train_index], instances[test_index]


class PairedInstancesTrainTestSplit(DataStrategy):
    """A train/test split policy for paired instances.

    The input data should have source_ids with two columns. This split assigns all sources to either the training set or
    the test set. The pairs are assigned to training or testing if both of their sources have that role. Pairs with
    mixed roles are omitted.
    """

    def __init__(self, test_size: float | int, seed: int | None = None):
        self.test_size = test_size
        self.seed = seed

    def apply(self, instances: FeatureData) -> Iterator[tuple[FeatureData, FeatureData]]:
        """Allow iteration by looping over the resulting train/test split(s)."""
        source_ids_1d = np.unique(check_not_none(instances.source_ids, 'missing field: `source_ids`'))
        sources_train, sources_test = sklearn.model_selection.train_test_split(
            source_ids_1d, test_size=self.test_size, shuffle=True, random_state=self.seed
        )

        sources_train = set(sources_train)

        def use_in_training(value: Any) -> bool:
            return value in sources_train

        training_source_ids = np.vectorize(use_in_training)(instances.source_ids)
        training_instances = np.all(training_source_ids, axis=1)
        test_instances = np.all(~training_source_ids, axis=1)

        yield instances[training_instances], instances[test_instances]


class RoleAssignment(Enum):
    """Indicate whether the data is part of the train or the test split."""

    TRAIN = 'train'
    TEST = 'test'


class PredefinedTrainTestSplit(DataStrategy):
    """
    Split data into a training set and a test set based on predefined assignments.

    This strategy expects a ``role_assignments`` field in the data, where each
    instance is labelled either ``"train"`` (included in the training set) or
    ``"test"`` (included in the test set).

    In the benchmark configuration YAML, this split strategy can be referenced as
    follows:

    .. code-block:: yaml

        cross_validation_splits:
            strategy: predefined_train_test_split
            data_origin: ${data}
    """

    def apply(self, instances: FeatureData) -> Iterable[tuple[FeatureData, FeatureData]]:
        """Split the FeatureData into a train and a test split."""
        if 'role_assignments' not in instances.all_fields:
            raise ValueError('`role_assignments` field is missing')

        training_set = instances[instances.role_assignments == RoleAssignment.TRAIN.value]  # type: ignore
        test_set = instances[instances.role_assignments == RoleAssignment.TEST.value]  # type: ignore
        yield training_set, test_set
