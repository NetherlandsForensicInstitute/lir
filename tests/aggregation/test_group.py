from pathlib import Path

import pytest

from lir import LLRData
from lir.aggregation import Aggregation, AggregationData
from lir.aggregation.group import AggregationGroup


@pytest.fixture
def dummy_data(synthesized_llrs: LLRData) -> AggregationData:
    return AggregationData(
        llrdata=synthesized_llrs,
        lrsystem=None,
        parameters={},
        experiment_output_dir=Path('/'),
        run_output_dir=Path('/'),
        run_name='test',
    )


class SingleAggregation(Aggregation):
    count_report: int = 0
    count_close: int = 0

    def report(self, data: AggregationData) -> None:
        self.count_report += 1

    def close(self):
        self.count_close += 1


def test_bad_arguments():
    with pytest.raises(ValueError):
        AggregationGroup(None)


def test_empty_list(dummy_data: AggregationData):
    group = AggregationGroup([])
    group.report(dummy_data)
    group.report(dummy_data)
    group.close()


def test_aggregation_group(dummy_data: AggregationData):
    a1 = SingleAggregation()
    a2 = SingleAggregation()
    group = AggregationGroup([a1, a2, AggregationGroup([a2])])

    for i in range(3):
        assert a1.count_report == i
        assert a2.count_report == i * 2
        group.report(dummy_data)

    for i in range(3):
        assert a1.count_close == i
        assert a2.count_close == i * 2
        group.close()
