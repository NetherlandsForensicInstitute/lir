import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Annotated, Any, Self, TypeVar

import numpy as np
from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator


LOG = logging.getLogger(__name__)


def _validate_labels(labels: np.ndarray | None) -> np.ndarray | None:
    """Check if labels have the correct shape."""
    if labels is None:
        return labels

    if len(labels.shape) != 1:
        raise ValueError(f'labels must be 1-dimensional; shape: {labels.shape}')

    if np.any((labels != 0) & (labels != 1)):
        raise ValueError(f'labels allowed: 0, 1; found: {np.unique(labels)}')

    return labels


class InstanceData(BaseModel, ABC):
    """
    Base class for data on instances.

    Attributes:
    - labels: an array of labels, a 1-dimensional array with one value per instance
    - source_ids: an array of source ids, a 2-dimensional array with one column and one value per instance
    """

    model_config = ConfigDict(frozen=True, extra='allow', arbitrary_types_allowed=True)

    labels: Annotated[np.ndarray | None, AfterValidator(_validate_labels)] = None
    source_ids: np.ndarray | None = None

    @model_validator(mode='after')
    def check_sourceids_labels_match(self) -> Self:
        if self.labels is not None and self.source_ids is not None and self.labels.shape[0] != self.source_ids.shape[0]:
            raise ValueError(
                f'dimensions of labels and source_ids do not match; "'
                f'{self.labels.shape[0]} != {self.source_ids.shape[0]}'
            )

        return self

    @model_validator(mode='after')
    def check_sourceid_shape(self) -> Self:
        if self.source_ids is None:
            return self

        # TODO: validate shape
        # if len(self.source_ids.shape) != 2 or self.source_ids.shape[1] != 1:
        #    raise ValueError(f'source_ids should be 2-dimensional with 1 column; found shape {self.source_ids.shape}')
        return self

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

    def __add__(self, other: 'InstanceData') -> Self:
        return self.combine(other, np.concatenate)

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

        if self.model_extra.keys() != other.model_extra.keys():  # type: ignore
            return False

        for field in self.all_fields:
            if type(getattr(self, field)) is not type(getattr(other, field)):
                return False

        return True

    def combine(
        self, others: 'list[InstanceData] | InstanceData', fn: Callable, *args: Any, **kwargs: dict[str, Any]
    ) -> Self:
        """
        Apply a custom combination function to InstanceData objects.

        All objects must have the same types and fields, and the same values for all non-numpy array
        fields, or an error is raised. Numpy fields are concatenated using `fn`. Other fields are copied as-is.
        """
        if isinstance(others, InstanceData):
            others = [others]

        # initialize the dictionary of fields to be updated
        data: dict[str, np.ndarray | None] = {}

        for field in self.all_fields:
            first_value = getattr(self, field)
            if isinstance(first_value, np.ndarray):
                # we have a numpy array field -> update required

                # collect values for all objects involved
                values = [first_value]
                for instances in others:
                    if not self.has_same_type(instances):
                        raise ValueError('instances to concatinate must have the same types and fields')
                    values.append(getattr(instances, field))

                # apply the function
                values = fn(values, *args, **kwargs)

                if field == 'labels' and len(values.shape) != 1:
                    # drop labels if they are in bad shape
                    data[field] = None
                else:
                    # store the value to be updated later
                    data[field] = values

            else:
                # we have a non-numpy array field -> check if they have the same value
                for instances in others:
                    other_value = getattr(instances, field)
                    if other_value != first_value:
                        raise ValueError(
                            f'unable to combine field `{field}`: value mismatch: {other_value} != {first_value}'
                        )

        return self.replace(**data)

    def apply(self, fn: Callable, *args: Any, **kwargs: dict[str, Any]) -> Self:
        """
        Apply a custom function to this InstanceData object.

        The function `fn` is applied to all Numpy fields. Other fields are copied as-is.
        """
        # initialize the dictionary of fields to be updated
        data: dict[str, np.ndarray | None] = {}

        for field in self.all_fields:
            values = getattr(self, field)
            if isinstance(values, np.ndarray):
                # we have a numpy array field -> update required

                if field == 'labels' and len(values.shape) != 1:
                    # drop labels if they are in bad shape
                    data[field] = None
                else:
                    # apply the function and store the value to be updated later
                    data[field] = fn(values, *args, **kwargs)

        return self.replace(**data)

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
        Returns a modified copy with updated values.

        :param kwargs: the fields to replace
        :return: the modified copy
        """
        return self.replace_as(type(self), **kwargs)

    def replace_as(self, datatype: type['InstanceDataType'], **kwargs: Any) -> 'InstanceDataType':
        """
        Returns a modified copy with updated data type and values.

        :param datatype: the return type
        :param kwargs: the fields to replace
        :return: the modified copy
        """
        args = self.model_dump()
        args.update(kwargs)
        return datatype(**args)


def _validate_features(features: np.ndarray) -> np.ndarray:
    """Check if labels have the correct shape."""
    if len(features.shape) < 2:
        LOG.debug(f'1d features are silently converted to 2d; found shape: {features.shape}')
        features = np.expand_dims(features, axis=1)

    return features


class FeatureData(InstanceData):
    """
    Data class for feature data.

    Attributes:
    - features: an array of instance features, with one row per instance
    """

    features: Annotated[np.ndarray, AfterValidator(_validate_features)]

    def __len__(self) -> int:
        return self.features.shape[0]

    @model_validator(mode='after')
    def check_matching_shapes(self) -> Self:
        if self.labels is not None and self.labels.shape[0] != self.features.shape[0]:
            raise ValueError(
                f'dimensions of labels and features do not match; {self.labels.shape[0]} != {self.features.shape[0]}'
            )
        if self.source_ids is not None and self.source_ids.shape[0] != self.features.shape[0]:
            raise ValueError(
                f'dimensions of source_ids and features do not match; "'
                f'{self.source_ids.shape[0]} != {self.features.shape[0]}'
            )
        return self

    @model_validator(mode='after')
    def check_features(self) -> Self:
        if not np.issubdtype(self.features.dtype, np.number):
            raise ValueError(f'features should be numeric; found: {self.features.dtype}')
        return self


class PairedFeatureData(FeatureData):
    """
    Data class for instance pair data.

    Attributes:
    - n_trace_instances: the number of trace instances in each pair
    - n_ref_instances: the number of reference instances in each pair
    - features: the features of all instances in the pair, with pairs along the first dimension, and instances along the
        second
    - source_ids: the source ids of the trace and reference instances of each pair, a 2-dimensional array with two
        columns
    - features_trace: the features of the trace instances
    - features_ref: the features of the reference instances
    - source_ids_trace: the source ids of the trace instances
    - source_ids_ref: the source ids of the reference instances
    """

    n_trace_instances: int
    n_ref_instances: int

    @property
    def features_trace(self) -> np.ndarray:
        return self.features[:, : self.n_trace_instances]

    @property
    def features_ref(self) -> np.ndarray:
        return self.features[:, self.n_trace_instances :]  # noqa: E203

    @property
    def source_ids_trace(self) -> np.ndarray | None:
        return self.source_ids[:, 0] if self.source_ids else None

    @property
    def source_ids_ref(self) -> np.ndarray | None:
        return self.source_ids[:, 1] if self.source_ids else None

    @model_validator(mode='after')
    def check_sourceid_shape(self) -> Self:
        """Overrides the `InstanceData` implementation."""
        if self.source_ids is not None and (len(self.source_ids.shape) != 2 or self.source_ids.shape[1] != 2):
            raise ValueError(f'source_ids should be 2-dimensional with 2 columns; found shape {self.source_ids.shape}')
        return self

    @model_validator(mode='after')
    def check_features_dimensions(self) -> Self:
        if len(self.features.shape) < 3:
            raise ValueError(f'features should have 3 or more dimensions; found shape: {self.features.shape}')
        if self.features.shape[1] != self.n_trace_instances + self.n_ref_instances:
            raise ValueError(
                f'features should have shape (*, {self.n_trace_instances}+{self.n_ref_instances}, *); '
                f'found: {self.features.shape[1]}'
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

    @model_validator(mode='after')
    def check_features_are_llrs(self) -> Self:
        if len(self.features.shape) > 2:
            raise ValueError(f'features must have 1 or 2 dimensions; shape: {self.features.shape}')
        if len(self.features.shape) == 2 and self.features.shape[1] != 3 and self.features.shape[1] != 1:
            raise ValueError(
                f'features must be 1-dimensional or 2-dimensional with 1 or 3 columns; shape: {self.features.shape}'
            )

        return self


InstanceDataType = TypeVar('InstanceDataType', bound=InstanceData)
FeatureDataType = TypeVar('FeatureDataType', bound=FeatureData)


def concatenate_instances(first: InstanceDataType, *others: InstanceDataType) -> InstanceDataType:
    """
    Concatenate the results of the InstanceData objects.

    All concatenated objects must have the same types and fields, and the same values for all non-numpy array fields,
    or an error is raised. Numpy fields are concatenated using `np.concatenate`. Other fields are copied as-is.
    """
    return first.combine(list(others), np.concatenate)


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
