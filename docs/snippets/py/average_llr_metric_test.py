from collections.abc import Iterable

from lir.aggregation import AggregationData

from .average_llr_metric import average_llr


def test_average_llr(aggregation_data: Iterable[AggregationData]):
    assert average_llr(aggregation_data[0].llrdata) == 0
    assert average_llr(aggregation_data[1].llrdata) == -1
    assert average_llr(aggregation_data[2].llrdata) == 2.5
