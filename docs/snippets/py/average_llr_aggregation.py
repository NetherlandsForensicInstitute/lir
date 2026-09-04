from pathlib import Path

import numpy as np

from lir.aggregation import Aggregation, AggregationData
from lir.config import ConfigValue, config_parser, pop_field


class AverageLLR(Aggregation):
    def __init__(self, path: Path):
        self.path = path
        self._average_cumulative = 0
        self._average_count = 0

    def report(self, data: AggregationData):
        average = np.average(data.llrdata.llrs)
        with open(self.path, 'a') as f:
            f.write(f'average LLR for run {data.run_name}: {average}\n')

        self._average_cumulative += average
        self._average_count += 1

    def close(self):
        with open(self.path, 'a') as f:
            f.write(f'overall average LLR: {self._average_cumulative / self._average_count}\n')


@config_parser
def average_llr(config: ConfigValue, output_dir: Path) -> AverageLLR:
    filename = pop_field(config, 'filename')
    return AverageLLR(output_dir / filename)
