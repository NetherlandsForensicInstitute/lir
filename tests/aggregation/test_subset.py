import tempfile
from pathlib import Path

import numpy as np

from lir import registry
from lir.aggregation import Aggregation, AggregationData, SubsetAggregation
from lir.data.models import LLRData
from lir.lrsystems.binary_lrsystem import BinaryLRSystem
from lir.transform import Identity


def test_subset_aggregation():
    class MyAggregation(Aggregation):
        def report(self, data: AggregationData) -> None:
            assert len(data.llrdata) == len(llrs) / 2, 'number of LLRs within a category'
            assert np.all(data.llrdata.llrs == data.llrdata.llrs[0]), (
                'LLRs of the same category must have the same value'
            )

    llrs = LLRData(features=np.arange(2).repeat(10).reshape((20, 1)), category=np.arange(2).repeat(10))
    aggregation = SubsetAggregation(aggregation_method=MyAggregation(), category_field='category')
    with tempfile.TemporaryDirectory() as experiment_output_dir:
        aggregation.report(
            AggregationData(
                run_name='testrun',
                llrdata=llrs,
                lrsystem=BinaryLRSystem(pipeline=Identity()),
                parameters={},
                experiment_output_dir=Path(experiment_output_dir),
                run_output_dir=Path(experiment_output_dir),
            )
        )


def test_subset_aggregation_registry():
    assert registry.get('output.by_category')
