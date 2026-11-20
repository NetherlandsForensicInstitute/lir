from collections.abc import Iterator

from lir import DataStrategy, InstanceData


class FullTrainTestSplit(DataStrategy):
    """
    Use the full data set for both training and testing.

    This strategy does not split the data. It yields a single train/test split in which both the training set and
    the test set contain all of the input instances.

    This strategy is useful when the data set is too small to make a meaningful train/test split, for example when
    assessing whether the data has any discriminative power at all. Note that performance metrics computed this way
    are optimistic, because the model is evaluated on data it has been trained on.

    In an experiment setup file, the split strategy can be referenced as:

    .. code-block:: yaml

        splits:
          strategy: train_test_full

    This strategy does not depend on hypothesis labels, source ids, or role assignments, and can therefore be used
    with any input data.
    """

    def apply[DataType: InstanceData](self, instances: DataType) -> Iterator[tuple[DataType, DataType]]:
        """
        Yield the full data set as both the training set and the test set.

        Parameters
        ----------
        instances : DataType
            Input instances to be processed by this method.

        Yields
        ------
        tuple[DataType, DataType]
            A single item, which is a tuple of the training set and the test set. Both sets contain all input
            instances.
        """
        yield instances, instances
