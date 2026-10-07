from pathlib import Path

from lir import DataProvider
from lir.aggregation import Aggregation, AggregationData
from lir.config import ConfigValue, config_parser
from lir.config.aggregation import parse_aggregations
from lir.config.base import ConfigAttribute
from lir.config.data import parse_data_provider
from lir.config.lrsystem_architectures import parse_lrsystem
from lir.experiments import Experiment
from lir.experiments.execution import LRSystemConfig
from lir.lrsystems import LRSystem


class InferenceExperiment(Experiment):
    """
    Experiment strategy that runs an inference session to apply a model on unlabeled data.

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
    lrsystem : LRSystem
        The LR system to train.
    output : Aggregation
        A method for aggregating results.
    inference_data : DataProvider
        Data provider that provides the data for training the LR system.
    training_data : DataProvider, optional
        Data provider that provides the data for training the LR system.
    """

    def __init__(
        self,
        output_dir: Path,
        lrsystem: LRSystem,
        output: Aggregation,
        inference_data: DataProvider,
        training_data: DataProvider | None = None,
    ):
        super().__init__(output_dir)
        self.lrsystem = lrsystem
        self.inference_data = inference_data
        self.output = output
        self.training_data = training_data

    def run(self) -> None:
        """Execute the experiment."""
        if self.training_data is not None:
            self.lrsystem.fit(self.training_data.get_instances())
        try:
            # Ensure the case data does not contain labels by setting them to None.
            instances = self.inference_data.get_instances().replace(hypothesis=None)

            llrs = self.lrsystem.apply(instances)
            aggregation_data = AggregationData(
                llrdata=llrs,
                lrsystem=self.lrsystem,
                parameters={},
                run_name='inference',
                experiment_output_dir=self.output_path,
                run_output_dir=self.output_path,
            )
            self.output.report(aggregation_data)

        finally:
            self.output.close()


@config_parser(
    attributes=[
        ConfigAttribute('lrsystem', LRSystemConfig, required=True),
        ConfigAttribute('output', list[Aggregation], required=True),
        ConfigAttribute('inference_data', DataProvider, required=True),
        ConfigAttribute('training_data', DataProvider, required=False),
    ]
)
def parse_inference_experiment(config: ConfigValue, output_dir: Path) -> InferenceExperiment:
    """
    Get an experiment for an inference session.

    Arguments:
    - lrsystem
    - output
    - inference_data
    - training_data

    Parameters
    ----------
    config : ConfigValue
        Experiment strategy configuration.
    output_dir : Path
        Base output directory for this experiment.

    Returns
    -------
    InferenceExperiment
        Inference session experiment.
    """
    with config:
        lrsystem = parse_lrsystem(config.pop('lrsystem'), output_dir)  # type: ignore
        aggregation = parse_aggregations(config.pop('output', validate_type=list), output_dir)
        inference_data = parse_data_provider(config.pop('inference_data'), output_dir)  # type: ignore

        training_data_config = config.pop('training_data', required=False)
        training_data = parse_data_provider(training_data_config, output_dir) if training_data_config else None

        return InferenceExperiment(output_dir, lrsystem, aggregation, inference_data, training_data)
