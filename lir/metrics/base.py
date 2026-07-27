import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any

from lir import LLRData
from lir.aggregation import AggregationData


LOG = logging.getLogger(__name__)


class MetricFunction(ABC):
    """Base class for a metric function that can be used with aggregations."""

    @abstractmethod
    def get_values(self, column_name: str, data: AggregationData) -> Iterable[tuple[str, str]]:
        """
        Get the values for a metric as a sequence of names and values.

        Parameters
        ----------
        column_name : str
            The user-provided name for the column.
        data : AggregationData
            An aggregation data object.

        Returns
        -------
        Iterable[tuple[str, str]]
            An iterable over metric names and values.
        """
        raise NotImplementedError


def llr_metric[ReturnType: Any](fn: Callable[[LLRData], ReturnType]) -> Callable[[LLRData], ReturnType]:
    """
    Decorate a metric function that can be used with aggregations.

    Parameters
    ----------
    fn : Callable[[LLRData], float]
        The decorated metric function.
    """

    class LLRMetric(MetricFunction):
        __doc__ = fn.__doc__

        def __call__(self, data: LLRData, *args: Any, **kwargs: Any) -> ReturnType:
            return fn(data, *args, **kwargs)  # type: ignore

        def get_values(self, column_name: str, data: AggregationData) -> Iterable[tuple[str, str]]:
            try:
                value = fn(data.llrdata)  # type: ignore
                if isinstance(value, (list, tuple)):
                    for index, metric_value in enumerate(value):
                        yield f'{column_name}_{index}', str(metric_value)
                else:
                    yield column_name, str(value)
            except Exception as e:
                LOG.warning(f'calculating metric {column_name} failed: {e}')
                yield column_name, ''

    return LLRMetric()


def aggregation_metric[ReturnType: Any](
    fn: Callable[[AggregationData], ReturnType],
) -> Callable[[AggregationData], ReturnType]:
    """
    Decorate a metric function that operates on ``AggregationData`` and can be used with aggregations.

    Parameters
    ----------
    fn : Callable[[AggregationData], float]
        The decorated metric function.
    """

    class AggregationMetric(MetricFunction):
        __doc__ = fn.__doc__

        def __call__(self, data: AggregationData, *args: Any, **kwargs: Any) -> ReturnType:
            return fn(data, *args, **kwargs)  # type: ignore

        def get_values(self, column_name: str, data: AggregationData) -> Iterable[tuple[str, str]]:
            try:
                yield column_name, str(fn(data))  # type: ignore
            except Exception as e:
                LOG.warning(f'obtaining value for {column_name} failed: {e}')
                yield column_name, ''

    return AggregationMetric()
