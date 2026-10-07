from lir.aggregation import Aggregation, AggregationData
from lir.util import check_type


class AggregationGroup(Aggregation):
    """
    Group of aggregation methods.

    Parameters
    ----------
    aggregations : list[Aggregation]
        Aggregations that belong to this group.
    """

    def __init__(self, aggregations: list[Aggregation]) -> None:
        self.aggregations = check_type(list, aggregations)

    def report(self, data: AggregationData) -> None:
        """
        Report that new results are available by calling the ``report()`` method on each aggregation in this group.

        Parameters
        ----------
        data : AggregationData
            The aggregated data to be reported.
        """
        for aggregation in self.aggregations:
            aggregation.report(data)

    def close(self) -> None:  # noqa: B027
        """
        Finalize the aggregation; no more results will come in.

        This calls the ``close()`` method on each aggregation in this group.
        """
        for aggregation in self.aggregations:
            aggregation.close()
