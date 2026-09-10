from collections.abc import Iterator

import numpy as np

from lir import DataStrategy, InstanceData
from lir.util import check_type


class LeaveOneCategoryOut(DataStrategy):
    """
    Split the data into a training set and a test set, following a leave-one-category-out scheme.

    This splitter assigns one category at a time to the test set, and the other categories to the training set. There
    are as many splits as there are categories.

    The ``category_field`` argument indicates which field in the input data contains the category of the instances.
    This field must be available in the input data, and it must be a 1-dimensional numpy array with the same length as
    the number of instances in the data.

    In an experiment setup file, the split strategy can be referenced as:

    .. code-block:: yaml

        splits:
          strategy: leave_one_category_out
          category_field: my_category

    Parameters
    ----------
    category_field : str
        The name of the field in the input data that contains the category of the instances.
    """

    def __init__(self, category_field: str):
        self.category_field = category_field

    def apply[DataType: InstanceData](self, instances: DataType) -> Iterator[tuple[DataType, DataType]]:
        """
        Split the data into a training set and a test set.

        One category at a time is assigned to the test set; the other categories to the training set. The number of
        splits is the number of categories.

        Parameters
        ----------
        instances : InstanceDataType
            Input instances.

        Yields
        ------
        tuple[DataType, DataType]
            An iterator over tuples of a training set and a test set.
        """
        if not hasattr(instances, self.category_field):
            raise ValueError(f'missing field in input data: {self.category_field}')

        category_values = check_type(np.ndarray, getattr(instances, self.category_field))
        if len(category_values.shape) != 1:
            raise ValueError(
                f'expected 1-dimensional array for category field {self.category_field}; '
                + 'found shape: {category_values.shape}'
            )

        for category in np.unique(category_values):
            yield instances[category_values != category], instances[category_values == category]
