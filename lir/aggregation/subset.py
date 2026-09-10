import logging
from collections.abc import Callable
from pathlib import Path

from lir.aggregation import Aggregation, AggregationData
from lir.data.models import get_instances_by_category


LOG = logging.getLogger(__name__)


class SubsetAggregation(Aggregation):
    """
    Aggregation method that manages data categorization.

    A separate aggregation method is used for each category.

    Parameters
    ----------
    aggregation_list_factory : Callable[[Path], list[Aggregation]]
        A factory that creates a list of aggregation methods for a category.
    category_field : str
        The name of the category field.
    """

    def __init__(self, aggregation_list: list[Aggregation], category_field: str):
        self.category_field = category_field
        self._aggregation_list = aggregation_list

    def report(self, data: AggregationData) -> None:
        """
        Report that new results are available.

        The data are categorized into subsets and forwarded to the actual aggregation method.

        Parameters
        ----------
        data : AggregationData
            The aggregated data to be reported.
        """
        run_name_prefix = f'{data.run_name}/' if data.run_name else ''
        for category, subset in get_instances_by_category(data.llrdata, self.category_field):
            category_str = '_'.join(str(v) for v in category.reshape(-1))
            run_name = f'{run_name_prefix}{category_str}'
            run_output_dir = data.experiment_output_dir / run_name
            category_data = AggregationData(
                llrdata=subset,
                lrsystem=data.lrsystem,
                parameters=data.parameters | {self.category_field: category_str},
                run_name=run_name,
                experiment_output_dir=data.experiment_output_dir,
                run_output_dir=run_output_dir,
                get_full_fit_lrsystem=data.get_full_fit_lrsystem,
            )

            # we need one set of aggregations per category
            # instantiate the aggregations if not already available
            for output in self._aggregation_list:
                output.report(category_data)

    def close(self) -> None:
        """Close all subset aggregation methods."""
        for output in self._aggregation_list:
            output.close()
