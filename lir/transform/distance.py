import numpy as np

from lir import Transformer
from lir.data.models import FeatureData, InstanceData, PairedFeatureData
from lir.util import check_type


class ElementWiseDifference(Transformer):
    """Calculate the element-wise absolute difference between pairs.

    Takes an array of sample pairs and returns the element-wise absolute difference.

    Expects:
    - a PairedFeatureData object with n_trace_instances=1 and n_ref_instances=1;

    Returns:
    - a copy of the FeatureData object with features of shape (n, f)
    """

    def apply(self, instances: InstanceData) -> FeatureData:
        """Calculate the absolute difference between all elements in the instance data (pairs)."""
        instances = check_type(PairedFeatureData, instances)
        if instances.n_ref_instances != 1 or instances.n_trace_instances != 1:
            raise ValueError(
                f'{self.__class__.__name__} must have exactly one reference instance and one trace instance;'
                f' found: n_ref_instances={instances.n_ref_instances}, n_trace_instances={instances.n_trace_instances}'
            )

        return instances.replace_as(FeatureData, features=np.abs(instances.features[:, 0] - instances.features[:, 1]))


class ManhattanDistance(Transformer):
    """Calculate the Manhattan distance between pairs.

    Takes a PairedFeatureData object or a FeatureData object and returns the manhattan distance.

    If the input is a PairedFeatureData object, the distance is computed as the manhattan distance, i.e. the sum of the
    element-wise difference between both sides of the pairs, for all features.

    If the input is a FeatureData object, it is assumed that it contains the element-wise differences, and the sum over
    these differences is calculated.

    :returns: a FeatureData object with features of shape (n, 1)
    """

    def apply(self, instances: InstanceData) -> FeatureData:
        """Calculate the Manhattan distance between all elements in the instance data (pairs)."""
        instances = check_type(FeatureData, instances)

        # if the data are paired instances, calculate the element wise difference first
        if isinstance(instances, PairedFeatureData):
            instances = ElementWiseDifference().apply(instances)

        # the feature axes are all axes except the first
        feature_axes = tuple(range(1, len(instances.features.shape)))

        # manhattan distance is the sum over all feature axes
        return instances.replace(features=np.sum(instances.features, axis=feature_axes))


class EuclideanDistance(Transformer):
    """Calculate the Euclidean distance between pairs.

    Takes a PairedFeatureData object or a FeatureData object and returns the euclidean distance.

    If the input is a PairedFeatureData object, the distance is computed as the euclidean distance, i.e. the square
    root of the sum of the squared element-wise difference between both sides of the pairs, for all features.

    If the input is a FeatureData object, it is assumed that it contains the element-wise differences, and the square
    root of the sum over these differences is calculated.

    :returns: a FeatureData object with features of shape (n, 1)
    """

    def apply(self, instances: InstanceData) -> FeatureData:
        """Calculate the Euclidean distance between all elements in the instance data (pairs)."""
        instances = check_type(FeatureData, instances)

        # if the data are paired instances, calculate the element wise difference first
        if isinstance(instances, PairedFeatureData):
            instances = ElementWiseDifference().apply(instances)

        # the feature axes are all axes except the first
        feature_axes = tuple(range(1, len(instances.features.shape)))

        # euclidean distance is the square root of the sum over all feature axes
        return instances.replace(features=np.sqrt(np.sum(instances.features**2, axis=feature_axes)))
