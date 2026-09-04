import tempfile
from pathlib import Path

import numpy as np

from lir import DataProvider, FeatureData
from lir.aggregation import AggregationData

from .data_counter_experiment import DataCounterExperiment


class GetData(DataProvider):
    def get_instances(self) -> FeatureData:
        return FeatureData(features=np.arange(12).reshape(6, 2))


def test_count_data(aggregation_data: list[AggregationData]):
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = DataCounterExperiment(data_provider=GetData(), output_dir=Path(tmpdir))
        exp.run()

        with open(Path(tmpdir) / 'counter.txt', 'r') as f:
            assert int(f.read()) == 6
