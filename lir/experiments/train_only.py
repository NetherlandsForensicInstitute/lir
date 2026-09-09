from pathlib import Path

from lir import DataProvider
from lir.config import ConfigValue, config_parser, pop_field
from lir.config.data import parse_data_provider
from lir.config.lrsystem_architectures import parse_lrsystem
from lir.experiments import Experiment
from lir.lrsystems import LRSystem
from lir.persistence import save_model


class TrainOnlyExperiment(Experiment):
    """
    Experiment strategy that runs a training session only.

    This experiment trains an LR system on a dataset and writes the fitted model to a file named ``lrsystem.pkl``.

    To set up a training only experiment in a YAML configuration:

    .. code-block:: yaml

        experiments:
          - strategy: train_only
            name: my experiment
            data_provider: *my_data_provider
            lrsystem: *my_lrsystem

    Parameters
    ----------
    output_dir : Path
        Path where generated outputs are written.
    data_provider : DataProvider
        Data provider that provides the data for training the LR system.
    lrsystem : LRSystem
        The LR system to train.
    """

    def __init__(self, output_dir: Path, data_provider: DataProvider, lrsystem: LRSystem):
        super().__init__(output_dir)
        self.data_provider = data_provider
        self.lrsystem = lrsystem

    def run(self) -> None:
        """Execute the experiment."""
        self.lrsystem.fit(self.data_provider.get_instances())
        save_model(self.output_path / 'lrsystem.pkl', self.lrsystem)


@config_parser
def parse_train_only_experiment(config: ConfigValue, output_dir: Path) -> TrainOnlyExperiment:
    """
    Get an experiment for a training session.

    Arguments:
    - data_provider
    - lrsystem

    Parameters
    ----------
    config : ConfigValue
        Experiment strategy configuration.
    output_dir : Path
        Base output directory for this experiment.

    Returns
    -------
    TrainOnlyExperiment
        Training session experiment.
    """
    with config:
        data_provider_config = pop_field(config, 'data_provider')
        data_provider = parse_data_provider(data_provider_config, output_dir)

        lrsystem_config = pop_field(config, 'lrsystem')
        lrsystem = parse_lrsystem(lrsystem_config, output_dir)

        return TrainOnlyExperiment(output_dir, data_provider, lrsystem)
