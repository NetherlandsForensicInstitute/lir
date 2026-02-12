from collections.abc import Iterator
from enum import Enum
from typing import Any

import numpy as np
import sklearn
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold

from lir.data.models import DataStrategy, FeatureData, InstanceDataType


class TrainTestSplit(DataStrategy):
    """
    Split the data into a training set and a test set.

    This splitter distributes the instances randomly over a training set and test set. Each instance is assigned to
    either the training set or the test set, but no sources will have instances that appear in both. The hypothesis
    labels are used to distribute the instances of each hypothesis proportionally to both sets.

    This splitter is suitable for most specific-source setups. If you have a common-source setup, take a look at
    ``SourcesTrainTestSplit``. Alternatively, use the ``CrossValidation`` strategy for cross-validation.

    In an experiment setup file, the split strategy can be referenced as:

    The input data should have hypothesis labels. This split assigns instances of both classes to the training set and
    the test set.
    """

    def __init__(self, test_size: float | int, seed: int | None = None):
        """
        Initialize the object.

        :param test_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include
            in the test split. If int, represents the absolute number of test samples.
        :param seed: The random state.
        """
        self.test_size = test_size
        self.seed = seed

    def apply(self, instances: FeatureData) -> Iterator[tuple[FeatureData, FeatureData]]:
        """
        Split the data into a training set and a test set.

        :return: an iterator over a single item, which is a tuple of the training set and the test set.
        """
        indexes = np.arange(len(instances))
        indexes_train, indexes_test = sklearn.model_selection.train_test_split(
            indexes, stratify=instances.require_labels, test_size=self.test_size, shuffle=True, random_state=self.seed
        )

        yield instances[indexes_train], instances[indexes_test]


class CrossValidation(DataStrategy):
    """
    K-fold cross-validation iterator over successive train/test splits.

    The input data must contain hypothesis labels. Each fold is constructed so that instances from both hypotheses are
    present in every split.

    This strategy may be registered in a YAML registry as follows:

    .. code-block:: yaml

        splits:
          strategy: cross_validation
          folds: 5  # the number k in k-fold cross-validation
          seed: 42  # optional

    """

    def __init__(self, folds: int, seed: int | None = None):
        """
        Initialize the object.

        :param folds: The number of train/test splits to return.
        :param seed: The random state.
        """
        self.folds = folds
        self.seed = seed
        self.shuffle = True if self.seed is not None else False  # noqa: SIM210

    def apply(self, instances: InstanceDataType) -> Iterator[tuple[InstanceDataType, InstanceDataType]]:
        """Return an iterator over *k* train/test splits."""
        kf = KFold(n_splits=self.folds, shuffle=self.shuffle, random_state=self.seed)
        for train_index, test_index in kf.split(np.arange(len(instances)), y=instances.labels):
            yield instances[train_index], instances[test_index]


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
    """

    def __init__(self, test_size: float | int, seed: int | None = None):
        """
        Class initialization.

        :param test_size: If float, should be between 0.0 and 1.0 and represent the proportion of sources to include in
            the test split (rounded up). If int, represents the absolute number of test sources.
        :param seed: The random state.
        """
        self.test_size = test_size
        self.seed = seed

    def apply(self, instances: InstanceDataType) -> Iterator[tuple[InstanceDataType, InstanceDataType]]:
        """
        Split the data into a training set and a test set.

        :return: an iterator over a single item, which is a tuple of the training set and the test set.
        """
        splitter = GroupShuffleSplit(test_size=self.test_size, n_splits=1, random_state=self.seed)
        ((train_index, test_index),) = splitter.split(np.arange(len(instances)), groups=instances.source_ids_1d)

        yield instances[train_index], instances[test_index]


class SourcesCrossValidation(DataStrategy):
    """
    K-fold cross-validation by source id.

    This data strategy uses the ``source_ids`` attribute and distributes the sources over *k* different subsets.
    Each of the subsets will be offered once as the test set, using the others as the training set. Each source is
    assigned to exactly one of the subsets, and no sources will have instances that appear in more than one.

    In an experiment setup file, the data strategy can be referenced as:

    .. code-block:: yaml

        splits:
          strategy: cross_validation_sources
          folds: 5

    This class internally uses ``sklearn.model_selection.GroupKFold``.
    """

    def __init__(self, folds: int):
        """:param folds: the number of train/test splits to return"""
        self.folds = folds

    def apply(self, instances: InstanceDataType) -> Iterator[tuple[InstanceDataType, InstanceDataType]]:
        """
        Perform *k*-fold cross-validation.

        Return an iterator over *k* train/test splits.
        """
        kf = GroupKFold(n_splits=self.folds)

        for train_index, test_index in kf.split(np.arange(len(instances)), groups=instances.source_ids_1d):
            yield instances[train_index], instances[test_index]


class PairsTrainTestSplit(DataStrategy):
    """A train/test split policy for paired instances.

    The input data should have ``source_ids`` with two columns. This split assigns all sources to either the training
    set or the test set. The pairs are assigned to training or testing if both of their sources have that role. Pairs
    with mixed roles are omitted.
    """

    def __init__(self, test_size: float | int, seed: int | None = None):
        """
        Initialize the object.

        :param test_size: The proportion of sources to include in the test set.
        :param seed: The random state.
        """
        self.test_size = test_size
        self.seed = seed

    def apply(self, instances: InstanceDataType) -> Iterator[tuple[InstanceDataType, InstanceDataType]]:
        """
        Split the data into a training set and a test set.

        :return: an iterator over a single item, which is a tuple of the training set and the test set.
        """
        source_ids = instances.source_ids
        if source_ids is None or len(source_ids.shape) != 2 or source_ids.shape[1] != 2:
            raise ValueError(f'expected two-column source_ids; shape found: {getattr(source_ids, "shape", None)}')
        sources_train, sources_test = sklearn.model_selection.train_test_split(
            np.unique(source_ids), test_size=self.test_size, shuffle=True, random_state=self.seed
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

    In the experiment setup file, this split strategy can be referenced as follows:

    .. code-block:: yaml

        cross_validation_splits:
            strategy: predefined_train_test
    """

    def apply(self, instances: InstanceDataType) -> Iterator[tuple[InstanceDataType, InstanceDataType]]:
        """
        Split the data into a training set and a test set.

        :return: an iterator over a single item, which is a tuple of the training set and the test set.
        """
        if 'role_assignments' not in instances.all_fields:
            raise ValueError('`role_assignments` field is missing')

        training_set = instances[instances.role_assignments == RoleAssignment.TRAIN.value]  # type: ignore
        test_set = instances[instances.role_assignments == RoleAssignment.TEST.value]  # type: ignore
        yield training_set, test_set
