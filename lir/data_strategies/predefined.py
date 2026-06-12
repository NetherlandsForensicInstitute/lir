from collections.abc import Iterator
from enum import Enum

import numpy as np

from lir import DataStrategy, InstanceData


class RoleAssignment(Enum):
    """Indicate whether the data is part of the train or the test split."""

    TRAIN = 'train'
    TEST = 'test'


def is_valid_input(instances: InstanceData) -> bool:  # numpydoc ignore=PR01,RT01
    """Return True iff predefined strategies can be applied."""
    return 'role_assignments' in instances.all_fields


def _check_input(instances: InstanceData) -> None:  # numpydoc ignore=PR01
    """Raise an error unless predefined strategies can be applied."""
    if 'role_assignments' not in instances.all_fields:
        raise ValueError('`role_assignments` field is missing')


class PredefinedTrainTestSplit(DataStrategy):
    """
    Split data into a training set and a test set based on predefined assignments.

    This strategy expects a ``role_assignments`` field in the data, where each
    instance is labelled either ``"train"`` (included in the training set) or
    ``"test"`` (included in the test set).

    In the experiment setup file, this split strategy can be referenced as follows:

    .. code-block:: yaml

        train_test_splits:
            strategy: predefined_train_test
    """

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

        training_set = instances[instances.role_assignments == RoleAssignment.TRAIN.value]  # type: ignore
        test_set = instances[instances.role_assignments == RoleAssignment.TEST.value]  # type: ignore
        yield training_set, test_set


class PredefinedCrossValidation(DataStrategy):
    """
    Split data into cross validation folds based on predefined assignments.

    This strategy expects a ``fold_assignments`` field in the data. For example, the
    ``parse_features_from_csv_file`` with the ``fold_assignment_column`` specifeid will create this field.

    Each instance should be labelled according in which test set (fold) the instance should be. This means that care
    should be taken to use the correct number of folds (= number of unique labels) and wether the folds are based on
    sources or on instances.

    In the experiment setup file, this split strategy can be referenced as follows:

    .. code-block:: yaml

        cross_validation_splits:
            strategy: predefined_cross_validation
    """

    def apply[DataType: InstanceData](self, instances: DataType) -> Iterator[tuple[DataType, DataType]]:
        """
        Perform cross-validation based on predefined fold assignments.

        This strategy expects a ``fold_assignments`` field in the data, where each instance is labelled with a fold
        identifier. The strategy will return one train/test split for each unique fold identifier, using the instances
        with that identifier as the test set and the others as the training set.

        Parameters
        ----------
        instances : InstanceDataType
            Input instances to be processed by this method.
        """
        _check_input(instances)

        fold_assignments = instances.fold_assignments  # type: ignore
        unique_folds = np.unique(fold_assignments)

        for fold in unique_folds:
            training_set = instances[fold_assignments != fold]  # type: ignore
            test_set = instances[fold_assignments == fold]  # type: ignore
            yield training_set, test_set
