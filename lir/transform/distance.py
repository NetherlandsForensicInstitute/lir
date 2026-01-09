import numpy as np

from lir import Transformer
from lir.data.models import FeatureData, InstanceData, PairedFeatureData
from lir.util import check_type


class ElementWiseDifference(Transformer):
    """
    Takes an array of sample pairs and returns the element-wise absolute difference.

    Expects:
        - a PairedFeatureData object with n_trace_instances=1 and n_ref_instances=1;
    Returns:
        - a copy of the FeatureData object with features of shape (n, f)
    """

    def transform(self, instances: InstanceData) -> FeatureData:
        instances = check_type(PairedFeatureData, instances)
        return instances.replace_as(
            FeatureData, features=np.abs(instances.features[:, 0, :] - instances.features[:, 1, :])
        )


class ManhattanDistance(Transformer):
    """
    Takes a PairedFeatureData object or a FeatureData object and returns the manhattan distance.

    Expects:
        - a FeatureData object; or
        - a PairedFeatureData object with n_trace_instances=1, n_ref_instances=1
    Returns:
        - a FeatureData object with features of shape (n, 1)
    """

    def transform(self, instances: InstanceData) -> FeatureData:
        instances = check_type(FeatureData, instances)
        if isinstance(instances, PairedFeatureData):
            instances = ElementWiseDifference().transform(instances)
        if len(instances.features.shape) != 2:
            raise ValueError(f'X must be of shape (n,f); found: {instances.features.shape}')

        return instances.replace(features=np.sum(instances.features, axis=1))
