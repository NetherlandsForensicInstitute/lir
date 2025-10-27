from abc import abstractmethod, ABC
from typing import Iterable, TypeVar, Self, Any, Annotated

import numpy as np
from pydantic import model_validator, AfterValidator, ConfigDict, BaseModel


def _validate_labels(labels: np.ndarray | None) -> np.ndarray | None:
    if labels is not None:
        if not isinstance(labels, np.ndarray):
            raise ValueError(f"labels must be None or np.ndarray; found: {type(labels)}")
        if len(labels.shape) != 1:
            raise ValueError(f"labels must be 1-dimensional; shape: {labels.shape}")

    return labels


class InstanceData(BaseModel, ABC):
    """
    Base class for data on instances.

    Attributes:
    - labels: an array of labels, a 1-dimensional array with one value per instance
    """

    model_config = ConfigDict(frozen=True, extra="allow", arbitrary_types_allowed=True)

    labels: Annotated[np.ndarray | None, AfterValidator(_validate_labels)] = None

    def __getitem__(self, indexes: np.ndarray) -> Self:
        data = {}
        for field in self.all_fields:
            values = getattr(self, field)
            if isinstance(values, np.ndarray):
                data[field] = values[indexes]
            else:
                data[field] = values
        return self.replace(**data)

    def __eq__(self, other: Any) -> bool:
        if type(self) is not type(other):
            return False

        for field in self.all_fields:
            value = getattr(self, field)
            other_value = getattr(other, field)
            if isinstance(value, np.ndarray) and isinstance(other_value, np.ndarray):
                if value.shape != other_value.shape or not np.all(value == other_value):
                    return False
            else:
                if value != other_value:
                    return False

        return True

    @property
    def all_fields(self) -> list[str]:
        all_fields = list(type(self).model_fields.keys())
        if self.model_extra:
            all_fields += list(self.model_extra.keys())
        return all_fields

    @property
    def has_labels(self) -> bool:
        return self.labels is not None

    def replace(self, **kwargs: Any) -> Self:
        args = self.model_dump()
        args.update(kwargs)
        return type(self)(**args)  # type: ignore


class FeatureData(InstanceData):
    """
    Data class for feature data.

    Attributes:
    - features: an array of instance features, with one row per instance
    """

    features: np.ndarray

    def __len__(self) -> int:
        return self.features.shape[0]

    @model_validator(mode="after")
    def check_matching_shapes(self) -> Self:
        if self.labels is not None and self.labels.shape[0] != self.features.shape[0]:
            raise ValueError(
                f"dimensions of labels and features do not match; {self.labels.shape[0]} != {self.features.shape[0]}"
            )
        return self


class LLRData(FeatureData):
    """
    Representation of calculated LLR values.
    """

    @property
    def llrs(self) -> np.ndarray:
        """
        :return: 1-dimensional numpy array of LLR values
        """
        if len(self.features.shape) == 1:
            return self.features
        else:
            return self.features[:, 0]

    @property
    def has_intervals(self) -> bool:
        """
        :return: indicate whether the LLR's have intervals
        """
        return len(self.features.shape) == 2 and self.features.shape[1] == 3

    @property
    def llr_intervals(self) -> np.ndarray | None:
        """
        :return: numpy array of LLR values of dimensions (n, 2), or `None` if the LLR's have no intervals
        """
        if self.has_intervals:
            return self.features[:, 1:]
        else:
            return None

    @model_validator(mode="after")
    def check_features_are_llrs(self) -> Self:
        if len(self.features.shape) > 2:
            raise ValueError(f"features must have 1 or 2 dimensions; shape: {self.features.shape}")
        if len(self.features.shape) == 2 and self.features.shape[1] != 3 and self.features.shape[1] != 1:
            raise ValueError(
                f"features must be 1-dimensional or 2-dimensional with 1 or 3 columns; shape: {self.features.shape}"
            )

        return self


InstanceDataType = TypeVar("InstanceDataType", bound=InstanceData)
FeatureDataType = TypeVar("FeatureDataType", bound=FeatureData)


def concatenate_instances(first: InstanceDataType, *others: InstanceDataType) -> InstanceDataType:
    """
    Combine the results of the InstanceData objects.
    """

    data = {}
    for field in first.all_fields:
        values = [getattr(first, field)]
        for instances in others:
            values.append(getattr(instances, field))
        if isinstance(values[0], np.ndarray):
            data[field] = np.concatenate(values)
        else:
            data[field] = values
    return first.replace(**data)


class DataSet(ABC):
    """General representation of a data source.

    Each data source should provide access to a feature vector, a target (ground truth labels) vector, and a meta data
    vector, by implementing the `get_instances()` method.
    """

    @abstractmethod
    def get_instances(self) -> FeatureData:
        """
        Returns a tuple of instances and their meta data. For a data set of `n`
        instances, this function returns a tuple of:

        - an array of instances with at least one dimension of size `n`;
        - an array of ground truth labels with dimensions `(n,)`;
        - an array of meta data with at least one dimension of size `n`.
        """

        raise NotImplementedError


class DataStrategy(ABC, Iterable):
    """
    General representation of a data setup strategy.

    All subclasses must implement a `__iter__()` method.

    The custom __iter__() method should return an iterator over tuples
    of a training set and a test set. Both the training set and the test
    set consist of a tuple of instances, labels, and meta data.
    """

    pass
