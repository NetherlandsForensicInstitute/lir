from typing import Self

import numpy as np

from lir import FeatureData, InstanceData, Transformer
from lir.transform import check_type


class ValidateFeatureDataType(Transformer):
    """
    Module that enforces the data types of the features in the instances.

    This transformer is useful for ensuring that the data types of the features in the instances are consistent with the
    data types determined during fitting. This can help prevent errors during the application of a model/pipeline.
    Especially useful when the model or the applied data is read from file.

    In short, it checks:
    1. That the number of features in the instances matches the number of data types determined during fitting.
    2. That the data types of the features can be cast to the data types determined during fitting.
    """

    _features_type: np.dtype
    _features_size: tuple[int, ...]

    def fit(self, instances: InstanceData) -> Self:
        """
        Fit the transformer to the data by determining the data types of each feature in the first instance.

        It assumes that all instances have the same data types for each feature.

        Parameters
        ----------
        instances : InstanceData
            The data to fit the transformer on.

        Returns
        -------
        Self
            The fitted transformer.
        """
        instances = check_type(FeatureData, instances)

        self._features_type = instances.features.dtype
        self._features_size = instances.features.shape[1:]

        return self

    def apply(self, instances: InstanceData) -> FeatureData:
        """
        Apply the transformer to the data by enforcing the data types of each feature.

        Will raise an error if the data types of the features in the instances do not match the data types determined
        during fitting.

        Parameters
        ----------
        instances : InstanceData
            The data to apply the transformer on.

        Returns
        -------
        FeatureData
            The transformed data with enforced data types.

        Raises
        ------
        ValueError
            If the number of features in the instances does not match the number of data types determined during fitting
        TypeError
            If the data types of the features in the instances cannot be cast to the data type determined during fitting
        """
        instances = check_type(FeatureData, instances)

        # Ensrure that the number of features matches the number of data types determined during fitting.
        if instances.features.shape[1:] != self._features_size:
            raise ValueError(f'Expected features of size {self._features_size} but got {instances.features.shape[1:]}')

        # Use np.can_cast to check if the value can be cast to the expected data type, which allows for some flexibility
        # in the data types (e.g., int can be cast to float, but not vice versa).
        if not np.can_cast(instances.features.dtype, self._features_type):
            raise TypeError(f'Expected features of type {self._features_type} but got {instances.features.dtype}')

        return instances
