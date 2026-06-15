import logging

from lir.aggregation import Aggregation, AggregationData
from lir.data.models import get_instances_by_category


LOG = logging.getLogger(__name__)


class SubsetAggregation(Aggregation):
    """
    Aggregation method that manages data categorization.

    A separate aggregation method is used for each category.

    Parameters
    ----------
    aggregation_methods : list[Aggregation]
        A list of methods to aggregate results by category.
    category_field : str
        The name of the category field.
    """

    def __init__(self, aggregation_methods: list[Aggregation], category_field: str):
        self.aggregation_methods = aggregation_methods
        self.category_field = category_field

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
            category_data = AggregationData(
                llrdata=subset,
                lrsystem=data.lrsystem,
                parameters=data.parameters | {self.category_field: str(category)},
                run_name=run_name,
                get_full_fit_lrsystem=data.get_full_fit_lrsystem,
            )

            for output in self.aggregation_methods:
                output.report(category_data)

    def close(self) -> None:
        """Close all subset aggregation methods."""
        for output in self.aggregation_methods:
            output.close()
