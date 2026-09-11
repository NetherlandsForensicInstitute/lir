import logging

from lir.aggregation import Aggregation, AggregationData
from lir.data.models import get_instances_by_category


LOG = logging.getLogger(__name__)


class SubsetAggregation(Aggregation):
    """
    Aggregation method that manages data categorization.

    This aggregation relies on a set of other aggregation methods which are called once for each category for each run.
    The `category_field` parameter refers to an attribute in the input data. It must be available as a numpy array of
    the same length as the number of instances, and its values are the categories of the instances.

    The :class:`~lir.aggregation.AggregationData` attributes are adjusted accordingly:

    - `llrdata` contains only instances of a single category.
    - `lrsystem` is unchanged.
    - `parameters` is modified to include the category field/value pair.
    - `run_name` is modified to include the category key/value pair, to make it unique within an experiment, even across
      categories.
    - `experiment_output_dir` is unchanged.
    - `run_output_dir` is modified to be unique within an experiment, even across categories.

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
        run_name_prefix = f'{data.run_name}_' if data.run_name else ''
        for category, subset in get_instances_by_category(data.llrdata, self.category_field):
            category_str = '_'.join(str(v) for v in category.reshape(-1))
            run_name = f'{run_name_prefix}{category_str}'

            category_data = AggregationData(
                llrdata=subset,
                lrsystem=data.lrsystem,
                parameters=data.parameters | {self.category_field: category_str},
                run_name=run_name,
                experiment_output_dir=data.experiment_output_dir,
                run_output_dir=data.run_output_dir.parent / f'{self.category_field}={category_str}',
                get_full_fit_lrsystem=data.get_full_fit_lrsystem,
            )

            for output in self.aggregation_methods:
                output.report(category_data)

    def close(self) -> None:
        """Close all subset aggregation methods."""
        for output in self.aggregation_methods:
            output.close()
