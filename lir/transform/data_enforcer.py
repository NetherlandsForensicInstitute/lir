from typing import Self

import numpy as np

from lir import FeatureData, InstanceData, Transformer
from lir.transform import check_type


class DataEnforcer(Transformer):
    """
    Module that enforces the data types of the features in the instances.

    This transformer is useful for ensuring that the data types of the features in the instances are consistent with the
    data types determined during fitting. This can help prevent errors during the application of a model/pipeline.
    Especially useful when the model or the applied data is read from file.
    """

    data_types: list[type]

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
        self.data_types = [type(instance) for instance in instances.features[0]]
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
        """
        instances = check_type(FeatureData, instances)

        # Ensrure that the number of features matches the number of data types determined during fitting.
        if instances.features.shape[1] != len(self.data_types):
            raise ValueError(f'Expected {len(self.data_types)} features but got {instances.features.shape[1]}')

        # Check that the data types of each feature in the instances match the data types determined during fitting.
        # Use np.can_cast to check if the value can be cast to the expected data type, which allows for some flexibility
        # in the data types (e.g., int can be cast to float, but not vice versa).
        for instance in instances.features:
            for value, data_type in zip(instance, self.data_types, strict=True):
                if not np.can_cast(value, data_type):
                    raise TypeError(f'Expected value of type {data_type} but got {type(value)} ({value}).')

        return instances
