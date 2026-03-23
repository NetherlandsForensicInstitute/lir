from typing import Self

from lir.data.models import FeatureData, InstanceData
from lir.transform import Transformer
from lir.util import check_type


class SaveFeatureTransformer(Transformer):
    """
    Transformer to save the features of the instances in a new field.

    The idea of this transformer is to capture the features of the instances at a specific point in the pipeline, and
    save them in a new field in the instance data. This can be useful for later analysis, plotting, or debugging.

    Parameters
    ----------
    save_as : str
        The name of the field to save the features in.
    """

    def __init__(self, save_as: str) -> None:
        self.save_as = save_as

    def fit(self, instances: InstanceData) -> Self:
        """
        Fit the transformer to the instances. This transformer does not require fitting, so this method returns self.

        Parameters
        ----------
        instances : InstanceData
            The instances to fit.

        Returns
        -------
        SaveFeatureTransformer
            The fitted transformer.
        """
        return self

    def apply(self, instances: InstanceData) -> InstanceData:
        """Save the features of the instances in a new field, and return the instances.

        Parameters
        ----------
        instances : InstanceData
            The instances to transform.

        Returns
        -------
        InstanceData
            The instances with the saved features.
        """
        instances = check_type(FeatureData, instances)
        return instances.replace(**{self.save_as: instances.features})
