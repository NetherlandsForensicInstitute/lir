import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from typing import Annotated, Any, Self, TypeVar

import numpy as np
from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator

from lir.util import check_type


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


def _validate_source_ids(source_ids: np.ndarray | None) -> np.ndarray | None:
    """Check if source_ids have the correct shape."""
    if source_ids is None:
        return source_ids

    if len(source_ids.shape) == 1:
        return source_ids

    # if we have a 2d array with one column, silently reshape it to 1d
    if len(source_ids.shape) == 2 and source_ids.shape[1] == 1:
        return source_ids.reshape(-1)

    if len(source_ids.shape) == 2 and source_ids.shape[1] == 2:
        return source_ids

    raise ValueError(
        f'source_ids must be either 1-dimensional or 2-dimensional with 2 columns; found shape {source_ids.shape}'
    )


class InstanceData(BaseModel, ABC):
    """
    Base class for data on instances.

    Attributes:
    - `labels`: The hypothesis labels of the instances, as a 1-dimensional array with one value per instance, can be
      either 0 or 1.
    - `source_ids`: The ids of all sources that contributed to the instances. Each instance is from a single source,
      except if it is a pair, in which case it has two sources. The source ids is either a 1-dimensional array or a
      2-dimensional array with two columns.
    """

    model_config = ConfigDict(frozen=True, extra='allow', arbitrary_types_allowed=True)

    labels: Annotated[np.ndarray | None, AfterValidator(_validate_labels)] = None
    source_ids: Annotated[np.ndarray | None, AfterValidator(_validate_source_ids)] = None

    @property
    def require_labels(self) -> np.ndarray:
        """Returns `labels` and guarantee that it is not None (or raise an error)."""
        if self.labels is None:
            raise ValueError('labels not set')
        return self.labels

    @model_validator(mode='after')
    def check_sourceids_labels_match(self) -> Self:
        """Validate the source_ids and labels have matching shapes."""
        if self.labels is not None and self.source_ids is not None and self.labels.shape[0] != self.source_ids.shape[0]:
            raise ValueError(
                f'dimensions of labels and source_ids do not match; "'
                f'{self.labels.shape[0]} != {self.source_ids.shape[0]}'
            )

        return self

    @property
    def source_ids_1d(self) -> np.ndarray:
        """:return: the attribute `source_ids` as a 1-dimensional array, with one source id per instance"""
        if self.source_ids is None:
            raise ValueError('source_ids not available')
        if len(self.source_ids.shape) != 1:
            raise ValueError(f'expected one source per instance; source_ids has illegal shape {self.source_ids.shape}')
        return self.source_ids

    @abstractmethod
    def __len__(self) -> int:
        """:return: the number of instances in this dataset"""
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
                # If indexes is an int, convert to array. This ensures the result is still an array, even if a single
                # index is provided.
                if isinstance(indexes, int):
                    indexes = np.array([indexes])
                data[field] = values[indexes]
            else:
                data[field] = values
        return self.replace(**data)

    def __add__(self, other: 'InstanceData') -> Self:
        return self.concatenate(other)

    def check_both_labels(self) -> np.ndarray:
        """
        Return labels or raise an error if they are missing or if they do not represent both hypotheses.

        :return: the labels
        :raise: ValueError if hypothesis labels are missing or either label is not represented.
        """
        if self.labels is None:
            raise ValueError('labels not set')
        if not np.all(np.unique(self.labels) == np.arange(2)):
            raise ValueError(f'not all classes are represented; labels found: {np.unique(self.labels)}')
        return self.labels

    @classmethod
    def _concatenate_field(cls, field: str, values: list[Any]) -> Any:
        if len(values) == 0:
            raise ValueError('no values to concatenate')

        if isinstance(values[0], np.ndarray):
            # we have a numpy array field -> use np.concatenate()
            return np.concatenate(values)

        # we have a non-numpy array field -> check if they have the same value
        all_equal = all(values[0] == other for other in values[1:])
        if not all_equal:
            raise ValueError(
                f'unable to combine field `{field}` because it is not a numpy array and not all values are equal'
            )

        # return the value, which is the same for all objects
        return values[0]

    def concatenate(self, *others: 'InstanceData') -> Self:
        """
        Concatenate instances from InstanceData objects.

        All concatenated objects must have the same types and fields. How fields are concatenated may depend on the
        subclass. By default, they must have the same values for all non-numpy array fields, or an error is raised.
        Numpy fields are concatenated using `np.concatenate`. Other fields are copied as-is.

        Returns a new object with the concatenated instances.
        """
        for instances in others:
            if not self.has_same_type(instances):
                raise ValueError('instances to concatenate must have the same types and fields')

        # initialize the dictionary of fields to be updated
        data: dict[str, np.ndarray | None] = {}

        for field in self.all_fields:
            all_values = [getattr(self, field)]
            all_values.extend([getattr(instances, field) for instances in others])
            data[field] = self._concatenate_field(field, all_values)

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

        if self.model_extra.keys() != other.model_extra.keys():  # type: ignore
            return False

        for field in self.all_fields:
            if type(getattr(self, field)) is not type(getattr(other, field)):
                return False

        return True

    def combine(self, others: 'list[InstanceData] | InstanceData', fn: Callable, *args: Any, **kwargs: Any) -> Self:
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
                        raise ValueError('instances to concatenate must have the same types and fields')
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

    def apply(self, fn: Callable, *args: Any, **kwargs: Any) -> Self:
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
        """:return: a list of all fields, including both mandatory and extra fields"""
        all_fields = list(type(self).model_fields.keys())
        if self.model_extra:
            all_fields += list(self.model_extra.keys())
        return all_fields

    @property
    def has_labels(self) -> bool:
        """:return: True iff the instances are labeled"""
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
        """Validate the shape of the features and the labels are matching."""
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
        """Validate the features."""
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
        """Get the features of the trace instances."""
        return self.features[:, : self.n_trace_instances]

    @property
    def features_ref(self) -> np.ndarray:
        """Get the features of the reference instances."""
        return self.features[:, self.n_trace_instances :]  # noqa: E203

    @property
    def source_ids_trace(self) -> np.ndarray | None:
        """Get the source ids of the trace instances."""
        return self.source_ids[:, 0] if self.source_ids else None

    @property
    def source_ids_ref(self) -> np.ndarray | None:
        """Get the source ids of the reference instances."""
        return self.source_ids[:, 1] if self.source_ids else None

    @model_validator(mode='after')
    def check_sourceid_shape(self) -> Self:
        """Overrides the `InstanceData` implementation."""
        if self.source_ids is not None and (len(self.source_ids.shape) != 2 or self.source_ids.shape[1] != 2):
            raise ValueError(f'source_ids should be 2-dimensional with 2 columns; found shape {self.source_ids.shape}')
        return self

    @model_validator(mode='after')
    def check_features_dimensions(self) -> Self:
        """Validate feature dimensions."""
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

    Attributes:
    - llrs: 1-dimensional numpy array of LLR values
    - has_intervals: indicate whether the LLR's have intervals
    - llr_intervals: numpy array of LLR values of dimensions (n, 2), or `None` if the LLR's have no intervals
    - llr_upper_bound: upper bound applied to the LLRs, or `None` if no upper bound was applied
    - llr_lower_bound: lower bound applied to the LLRs, or `None` if no lower bound was applied

    """

    llr_upper_bound: float | None = None
    llr_lower_bound: float | None = None

    @property
    def llrs(self) -> np.ndarray:
        """:return: 1-dimensional numpy array of LLR values"""
        if len(self.features.shape) == 1:
            return self.features
        else:
            return self.features[:, 0]

    @property
    def has_intervals(self) -> bool:
        """:return: indicate whether the LLR's have intervals"""
        return len(self.features.shape) == 2 and self.features.shape[1] == 3

    @property
    def llr_intervals(self) -> np.ndarray | None:
        """:return: numpy array of LLR values of dimensions (n, 2), or `None` if the LLR's have no intervals"""
        if self.has_intervals:
            return self.features[:, 1:]
        else:
            return None

    @property
    def llr_bounds(self) -> tuple[float | None, float | None]:
        """:return: a tuple (min_llr, max_llr)"""
        return self.llr_lower_bound, self.llr_upper_bound

    @model_validator(mode='after')
    def check_features_are_llrs(self) -> Self:
        """Validate the feature data."""
        if len(self.features.shape) > 2:
            raise ValueError(f'features must have 1 or 2 dimensions; shape: {self.features.shape}')

        if len(self.features.shape) == 2 and self.features.shape[1] != 3 and self.features.shape[1] != 1:
            raise ValueError(
                f'features must be 1-dimensional or 2-dimensional with 1 or 3 columns; shape: {self.features.shape}'
            )

        if self.has_intervals and (
            np.all(self.features[:, 1] > self.features[:, 0]) or np.all(self.features[:, 2] < self.features[:, 0])
        ):
            raise ValueError('LLRs should not exceed their own intervals')

        return self

    @classmethod
    def _concatenate_field(cls, field: str, values: list[Any]) -> Any:
        """Remove `llr_upper_bound` and `llr_lower_bound` when having different values.

        The fields `llr_upper_bound` and `llr_lower_bound` may have different values which is not allowed by default.
        Remove them instead of trying to combine them.
        """
        match field:
            case 'llr_upper_bound' | 'llr_lower_bound':
                # Check if all values are the same; if so, preserve the value
                if all(v == values[0] for v in values):
                    return values[0]
                # Otherwise, return None when values differ
                return None
            case _:
                return super()._concatenate_field(field, values)

    def check_misleading_finite(self) -> None:
        """Check whether all values are either finite or not misleading."""
        values, labels = self.llrs, self.require_labels

        # give error message if H1's contain zeros and H2's contain ones
        if np.any(np.isneginf(values[labels == 1])) and np.any(np.isposinf(values[labels == 0])):
            raise ValueError('invalid input: -inf found for H1 and inf found for H2')
        # give error message if H1's contain zeros
        if np.any(np.isneginf(values[labels == 1])):
            raise ValueError('invalid input: -inf found for H1')
        # give error message if H2's contain ones
        if np.any(np.isposinf(values[labels == 0])):
            raise ValueError('invalid input: inf found for H2')


InstanceDataType = TypeVar('InstanceDataType', bound=InstanceData)
FeatureDataType = TypeVar('FeatureDataType', bound=FeatureData)


def concatenate_instances(first: InstanceDataType, *others: InstanceDataType) -> InstanceDataType:
    """Concatenate the results of the InstanceData objects.

    Alias for `first.concatenate(*others)`.
    """
    return first.concatenate(*others)


class DataProvider(ABC):
    """Base class for data providers.

    Each data provider should provide access to instance data by implementing the `get_instances()` method.
    """

    @abstractmethod
    def get_instances(self) -> FeatureData:
        """Returns an InstanceData object, containing data for a set of instances."""
        raise NotImplementedError


class DataStrategy(ABC):
    """Base class for data (splitting) strategies."""

    @abstractmethod
    def apply(self, instances: FeatureData) -> Iterable[tuple[FeatureData, FeatureData]]:
        """Provide iterator to access training and test set.

        Returns an iterator over tuples of a training set and a test set. Both the training set and the test
        is represented by an `InstanceData` object.
        """
        raise NotImplementedError


def get_instances_by_category[InstanceDataType: InstanceData](
    instances: InstanceDataType, category_field: str, category_shape: tuple[int] | None = None
) -> Iterator[tuple[np.ndarray, InstanceDataType]]:
    """
    Return subsets of a set of instances by category.

    The `instances` object must have a field by the name of `category_field`. That field is a numpy array with one row
    per instance. Its values are the categories of each instance. The field may have any shape, as long as the number of
    rows matches the number of instances.

    If `category_shape` is provided, the shape of the category field is checked against this value.

    The returned value is an iterator with each item being a tuple of the category and the subset of instances of that
    category.

    :param instances: the set of instances to draw from
    :param category_field: the name of the field in instances that indicates the categories
    :param category_shape: the optional shape of the category field
    :return: tuples of categories and corresponding subsets of instances
    """
    # extract the category values from the instances
    if not hasattr(instances, category_field):
        raise ValueError(f'missing field: {category_field}')
    category_values = getattr(instances, category_field)

    # check the category values for sanity
    check_type(np.ndarray, category_values, 'categories must be a numpy array')
    if category_values.shape[0] != len(instances):
        raise ValueError(
            f'number of categories does not equal number of instances: {category_values.shape[0]} != {len(instances)}'
        )

    # check for shape, if available
    if category_shape is not None:
        expected_category_shape = (len(instances),) + category_shape
        if category_values.shape != expected_category_shape:
            raise ValueError(
                f'expected shape of category field {category_field}: {expected_category_shape}; '
                f'found: {category_values.shape}'
            )

    # each unique value is a category
    unique_values = np.unique(category_values, axis=0)

    # return the subset of instances for each category separately
    for value in unique_values:
        current_category_rows = np.all(category_values == value, axis=tuple(range(1, category_values.ndim)))
        yield value, instances[current_category_rows]
