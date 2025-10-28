from abc import abstractmethod, ABC
from typing import Iterable, TypeVar, Self, Any, Annotated

import numpy as np
from pydantic import model_validator, AfterValidator, ConfigDict, BaseModel


def _validate_labels(labels: np.ndarray | None) -> np.ndarray | None:
    """Check if labels have the correct shape."""
    if labels is None or len(labels.shape) == 1:
        return labels

    raise ValueError(f"labels must be 1-dimensional; shape: {labels.shape}")


class InstanceData(BaseModel, ABC):
    """
    Base class for data on instances.

    Attributes:
    - labels: an array of labels, a 1-dimensional array with one value per instance
    """

    model_config = ConfigDict(frozen=True, extra="allow", arbitrary_types_allowed=True)

    labels: Annotated[np.ndarray | None, AfterValidator(_validate_labels)] = None

    @abstractmethod
    def __len__(self) -> int:
        """
        :return: the number of instances in this dataset
        """
        raise NotImplementedError

    def __getitem__(self, indexes: np.ndarray | int) -> Self:
        """
        Get a copy of a subset of instances.

        All `ndarray` fields are indexed using `indexes`.
        All other fields are taken as-is.

        :param indexes: the indexes to select
        :return: a subset of this dataset
        """
        data = {}
        for field in self.all_fields:
            values = getattr(self, field)
            if isinstance(values, np.ndarray):
                data[field] = values[indexes]
            else:
                data[field] = values
        return self.replace(**data)

    def has_same_type(self, other: Any) -> bool:
        """
        Compare these instance data to another class.

        Returns True iff:
        - `other` has the same class
        - `other` has the same fields
        - all fields have the same type
        """
        if type(self) is not type(other):
            return False

        if self.model_extra.keys() != other.model_extra.keys():
            return False

        for field in self.all_fields:
            if type(getattr(self, field)) != type(getattr(other, field)):
                return False

        return True


    def __eq__(self, other: Any) -> bool:
        """
        Compare these instance data to another class.

        Returns True iff:
        - the method `has_same_type()` returns `True`
        - all numpy fields in `other` have the same shape and the same values
        - all other fields are compared using the `!=` operator
        """
        if not self.has_same_type(other):
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
        """
        :return: a list of all fields, including both mandatory and extra fields
        """
        all_fields = list(type(self).model_fields.keys())
        if self.model_extra:
            all_fields += list(self.model_extra.keys())
        return all_fields

    @property
    def has_labels(self) -> bool:
        """
        :return: True iff the instances are labeled
        """
        return self.labels is not None

    def replace(self, **kwargs: Any) -> Self:
        """
        :return: a copy of these instances, while replacing specific fields
        """
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

    All concatenated objects must have the same types and fields, and the same values for all non-numpy array fields,
    or an error is raised. Numpy fields are concatenated using `np.concatenate`. Other fields are copied as-is.
    """

    data = {}
    for field in first.all_fields:
        first_value = getattr(first, field)
        if isinstance(first_value, np.ndarray):
            # we have a numpy array field -> concatenate
            values = [first_value]

            for instances in others:
                if not first.has_same_type(instances):
                    raise ValueError("instances to concatinate must have the same types and fields")
                values.append(getattr(instances, field))

            data[field] = np.concatenate(values)
        else:
            # we have another field -> check if they have the same value
            for instances in others:
                other_value = getattr(instances, field)
                if other_value != first_value:
                    raise ValueError(f"unable to concatenate field `{field}`: value mismatch: {other_value} != {first_value}")

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
