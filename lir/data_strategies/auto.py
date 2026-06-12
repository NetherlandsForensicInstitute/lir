from collections.abc import Iterator

from lir import DataStrategy, InstanceData
from lir.data_strategies import labels, pairs, predefined, sources


class AutoTrainTestSplit(DataStrategy):
    """
    Split the data into a training set and a test set.

    This splitter attempts to find a suitable splitting strategy for the input data. Candidate strategies are, in order
    of priority:

    - :class:`~lir.data_strategies.PredefinedTrainTestSplit`, if the dataset has role assignments;
    - :class:`~lir.data_strategies.PairsTrainTestSplit`, if the dataset has pairs with source ids (i.e., two source ids
      per pair);
    - :class:`~lir.data_strategies.SourcesTrainTestSplit`, if the instances in the dataset have source ids;
    - :class:`~lir.data_strategies.TrainTestSplit`, if the instances in the dataset have hypothesis labels.

    The data strategy will be decided in ``apply()``. Subsequent calls to ``apply()`` are not guaranteed to use the same
    strategy, although in realistic use cases this is most likely the case. If no suitable strategy is found, a
    :class:`ValueError` is raised.

    In an experiment setup file, the split strategy can be referenced as:

    .. code-block:: yaml

        splits:
          strategy: auto_train_test
          test_size: 0.2  # the (hold-out) test set  is 20% of the data
          seed: 42  # optional

    Parameters
    ----------
    test_size : float | int
        Size of the test set. If `float`, should be between 0.0 and 1.0 and represent the proportion of the dataset to
        include in the test split. If `int`, represents the absolute number of test samples. The default value is 0.5.
    random_state : int | None
        Random seed controlling stochastic behaviour for reproducible results.
    """

    def __init__(self, test_size: float | int = 0.5, random_state: int | None = None):
        self.test_size = test_size
        self.random_state = random_state

    def apply[DataType: InstanceData](self, instances: DataType) -> Iterator[tuple[DataType, DataType]]:
        """
        Split the data into a training set and a test set.

        Parameters
        ----------
        instances : DataType
            Instances to split.

        Returns
        -------
        Iterator[tuple[DataType, DataType]]
            An iterator over pairs of a training set and a test set, which is a tuple of the training set and the test
            set.
        """
        strategy: DataStrategy
        if predefined.is_valid_input(instances):
            strategy = predefined.PredefinedTrainTestSplit()
        elif pairs.is_valid_input(instances):
            strategy = pairs.PairsTrainTestSplit(self.test_size, self.random_state)
        elif sources.is_valid_input(instances):
            strategy = sources.SourcesTrainTestSplit(self.test_size, self.random_state)
        elif labels.is_valid_input(instances):
            strategy = labels.TrainTestSplit(self.test_size, self.random_state)
        else:
            raise ValueError('no valid data strategy found for the input data')

        return strategy.apply(instances)


class AutoCrossValidation(DataStrategy):
    """
    K-fold cross-validation iterator over successive train/test splits.

    This splitter attempts to find a suitable splitting strategy for the input data. Candidate strategies are:

    - :class:`~lir.data_strategies.PredefinedCrossValidation`, if the dataset has role assignments;
    - :class:`~lir.data_strategies.SourcesCrossValidation`, if the dataset has pairs with source ids (i.e., two source
      ids per pair);
    - :class:`~lir.data_strategies.CrossValidation`, if the instances in the dataset have hypothesis labels.

    The data strategy will be decided in ``apply()``. Subsequent calls to ``apply()`` are not guaranteed to use the same
    strategy, although in realistic use cases this is most likely the case. If no suitable strategy is found, a
    :class:`ValueError` is raised.

    This strategy may be referenced in a YAML setup as follows:

    .. code-block:: yaml

        splits:
          strategy: auto_cross_validation
          folds: 5  # the number k in k-fold cross-validation
          random_state: 42  # optional

    Parameters
    ----------
    folds : int
        Number of cross-validation folds to generate.
    shuffle : bool | None
        Whether to shuffle the groups before splitting into batches. If `None`, the data will be shuffled if
        `random_state` is not `None`.
    random_state : int | None
        Random seed controlling stochastic behavior for reproducible results.
    """

    def __init__(self, folds: int, shuffle: bool | None = None, random_state: int | None = None):
        self.folds = folds
        self.random_state = random_state
        self.shuffle: bool = shuffle or (shuffle is None and random_state is not None)

    def apply[DataType: InstanceData](self, instances: DataType) -> Iterator[tuple[DataType, DataType]]:
        """
        Return an iterator over *k* train/test splits.

        Parameters
        ----------
        instances : InstanceDataType
            Input instances to be processed by this method.

        Returns
        -------
        Iterator[tuple[DataType, DataType]]
            An iterator over pairs of a training set and a test set.
        """
        strategy: DataStrategy
        if predefined.is_valid_input(instances):
            strategy = predefined.PredefinedCrossValidation()
        elif sources.is_valid_input(instances):
            strategy = sources.SourcesCrossValidation(
                folds=self.folds, shuffle=self.shuffle, random_state=self.random_state
            )
        elif labels.is_valid_input(instances):
            strategy = labels.CrossValidation(folds=self.folds, shuffle=self.shuffle, seed=self.random_state)
        else:
            raise ValueError('no valid data strategy found for the input data')

        return strategy.apply(instances)
