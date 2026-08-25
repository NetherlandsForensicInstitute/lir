from pathlib import Path

from lir import DataProvider
from lir.config import ConfigValue, config_parser, pop_field
from lir.config.data import parse_data_provider
from lir.experiments import Experiment


class DataCounterExperiment(Experiment):
    def __init__(self, data_provider: DataProvider, output_dir: Path):
        super().__init__(output_path=output_dir)
        self.data_provider = data_provider

    def run(self):
        number_of_instances = len(self.data_provider.get_instances())
        with open(self.output_path / 'counter.txt', 'w') as f:
            f.write(f'{number_of_instances}\n')


@config_parser
def parse_data_counter_experiment_config(config: ConfigValue, output_dir: Path) -> Experiment:
    data_provider_config = pop_field(config, 'data_provider')
    data_provider = parse_data_provider(data_provider_config, output_dir)
    return DataCounterExperiment(data_provider, output_dir)
