from collections.abc import Iterator
from enum import Enum
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold

from lir.config.base import ContextAwareDict, check_is_empty, config_parser, pop_field
from lir.config.data_providers import parse_data_provider
from lir.data.models import DataSet, DataStrategy, FeatureData


class BinaryTrainTestSplit(DataStrategy):
    """Representation of a train/test split.

    The input data should have class labels. This split assigns instances of both classes to the training set and the
    test set.
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

    The input data should have class labels. This split assigns instances of both classes to each "fold" subset.
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

    The input data should have source_ids. This split assigns all instances of a source to either the training set or
    the test set.
    """

    def __init__(self, source: DataSet, test_size: float | int, seed: int | None = None):
        self.source = source
        self.test_size = test_size
        self.seed = seed

    def __iter__(self) -> Iterator:
        """Allow iteration by looping over the resulting train/test split(s)."""
        splitter = GroupShuffleSplit(test_size=self.test_size, n_splits=1, random_state=self.seed)
        instances = self.source.get_instances()
        ((train_index, test_index),) = splitter.split(instances.features, instances.source_ids, instances.source_ids)

        yield instances[train_index], instances[test_index]


class MulticlassCrossValidation(DataStrategy):
    """Representation of a K-fold cross validation iterator over each train/test split fold.

    The input data should have source_ids. This split assigns all instances of a source to the same "fold" subset.
    """

    def __init__(self, source: DataSet, folds: int):
        self.source = source
        self.folds = folds

    def __iter__(self) -> Iterator:
        """Allow iteration by looping over the resulting train/test split(s)."""
        kf = GroupKFold(n_splits=self.folds)

        instances = self.source.get_instances()
        for _i, (train_index, test_index) in enumerate(kf.split(instances.features, groups=instances.source_ids)):
            yield instances[train_index], instances[test_index]


class RoleAssignment(Enum):
    TRAIN = 'train'
    TEST = 'test'


class PredefinedTrainTestSplit(DataStrategy):
    """
    Splits data into a training set and a test set, according to pre-existing assignments in the data.

    Presumes a `role_assignments` field in the data, which has the value "train" for instances that will be part of the
    training set, and "test" for instances in the test set.
    """

    def __init__(self, data_provider: DataSet):
        self.data_provider = data_provider

    def __iter__(self) -> Iterator[tuple[FeatureData, FeatureData]]:
        instances = self.data_provider.get_instances()
        if 'role_assignments' not in instances.all_fields:
            raise ValueError('`role_assignments` field is missing')

        training_set = instances[instances.role_assignments == RoleAssignment.TRAIN.value]  # type: ignore
        test_set = instances[instances.role_assignments == RoleAssignment.TEST.value]  # type: ignore
        yield training_set, test_set


@config_parser
def predefined_train_test_split(config: ContextAwareDict, output_path: Path) -> DataStrategy:
    """
    Initialize a train/test splitter, PredefinedTrainTestSplitter.

    In the benchmark configuration YAML, this validation can be referenced as follows:
    ```
    cross_validation_splits:
        strategy: predefined_train_test_split
        data_origin: ${data}
    ```
    """
    data_provider = parse_data_provider(pop_field(config, 'data_origin'), output_path)
    check_is_empty(config)
    return PredefinedTrainTestSplit(data_provider)
