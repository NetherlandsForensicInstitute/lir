import tempfile
from pathlib import Path

from lir.aggregation import AggregationData

from .average_llr_aggregation import AverageLLR


def test_average_llr(aggregation_data: list[AggregationData]):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'avg.txt'
        aggregation = AverageLLR(path)

        for data in aggregation_data:
            aggregation.report(data)
        aggregation.close()

        with open(path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        for i, data in enumerate(aggregation_data):
            assert lines[i] == f'average LLR for run {data.run_name}: {data.llrdata.average_llr}'
        assert lines[3] == 'overall average LLR: 0.5'
