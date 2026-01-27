import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import confidence
from tqdm import tqdm

import lir
from lir.aggregation import Aggregation, AggregationData
from lir.config.base import ContextAwareDict
from lir.config.data import parse_data_object
from lir.config.lrsystem_architectures import parse_lrsystem
from lir.config.util import simplify_data_structure
from lir.data.models import InstanceData, concatenate_instances
from lir.lrsystems.lrsystems import LLRData


LOG = logging.getLogger(__name__)


class Experiment(ABC):
    """Representation of an experiment pipeline run for each provided LR system."""

    def __init__(
        self,
        name: str,
        outputs: Sequence[Aggregation],
        output_path: Path,
    ):
        self.name = name
        self.outputs = outputs
        self.output_path = output_path

    def _run_lrsystem(
        self,
        lrsystem_config: ContextAwareDict,
        split_data: Iterable[tuple[InstanceData, InstanceData]],
        parameters: dict[str, Any],
        run_name: str,
        data_config: ContextAwareDict,
    ) -> LLRData:
        """Run experiment on a single LR system configuration using the provided data(setup).

        First, the data is split into a training and testing subset, according to the provided
        data setup strategy. Next, the LR system is fitted using the training subset data and
        subsequently used to determine LLRs for the test subset data. The results are stored in
        a temporary list which contains the determined data of each test / train split.

        The collected results are combined and passed to the configured `outputs` aggregations,
        which may write metrics and visualizations to the `output_path` directory. The combined
        LLR data is returned.
        """
        # write the configuration to the output folder (data and lrsystem)
        output_dir = self.output_path / run_name
        lrsystem = parse_lrsystem(deepcopy(lrsystem_config), output_dir)

        config_dict = {
            'lr_system': simplify_data_structure(lrsystem.config),
            'data': simplify_data_structure(data_config),
        }

        output_config = confidence.Configuration(config_dict)
        output_dir.mkdir(exist_ok=True, parents=True)
        confidence.dumpf(output_config, output_dir / 'configuration.yaml')

        # Placeholders for numpy arrays of LLRs and labels obtained from each train/test split
        llr_sets: list[LLRData] = []

        # Split the data into a train / test subset, according to the provided DataStrategy. This could
        # for example be a simple binary split or a multiple fold cross validation split.
        for training_data, test_data in split_data:
            lrsystem.fit(training_data)
            subset_llr_results: LLRData = lrsystem.apply(test_data)

            # Store results (numpy arrays) into the placeholder lists
            llr_sets.append(subset_llr_results)

        # Combine collected numpy arrays after iteration over the train/test split(s)
        combined_llrs: LLRData = concatenate_instances(*llr_sets)

        # Collect and report results as configured by `outputs`
        results = AggregationData(
            llrdata=combined_llrs, lrsystem=lrsystem, parameters=parameters, run_name=run_name
        )
        for output in self.outputs:
            output.report(results)

        return combined_llrs

    @abstractmethod
    def _generate_and_run(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        """Run experiment for all configured LR systems.

        Perform the single experiment of all configured LR systems and write the obtained
        key metrics - results on the performance of the LR system - to the dedicated `metrics.csv`
        file in the `output_path` directory.
        """
        try:
            self._generate_and_run()
        finally:
            for output in self.outputs:
                output.close()


class PredefinedExperiment(Experiment):
    """Representation of an experiment run for each provided LR system."""

    def __init__(
        self,
        name: str,
        data_configs: list[tuple[ContextAwareDict, dict[str, Any]]],
        outputs: Sequence[Aggregation],
        output_path: Path,
        lrsystems: list[tuple[ContextAwareDict, dict[str, Any]]],
    ):
        super().__init__(name, outputs, output_path)
        self.lrsystem_config = lrsystems
        self.data_configs = data_configs

    def _generate_and_run(self) -> None:
        # Only display the data configuration progress bar when running interactively and
        # there are multiple data configurations to evaluate.
        disable_data_tqdm = not lir.is_interactive() or len(self.data_configs) == 1

        for data_config, dataparameter in tqdm(self.data_configs, disable=disable_data_tqdm, desc=self.name):
            # Parse the data configuration. This is done here to ensure that data
            # parsing is only done once per data configuration, even when multiple
            # LR systems are being evaluated on the same data setup.
            provider, splitter = parse_data_object(deepcopy(data_config), self.output_path)
            split_data = list(splitter.apply(provider.get_instances()))

            # Only display the LR system configuration progress bar when running interactively, and
            # the data_tqdm is not disabled, and there are multiple LR systems to evaluate.
            disable_lrsystem_tqdm = not lir.is_interactive() or not disable_data_tqdm or len(self.lrsystem_config) == 1
            for lrsystem_config, hyperparameters in tqdm(
                self.lrsystem_config, desc=self.name, disable=disable_lrsystem_tqdm
            ):
                # Combine the data parameter with the LR system hyperparameters to create
                # a unique name for this experiment configuration.
                data_name = '__'.join([f'{key}={value}' for key, value in dataparameter.items()])
                lrsystem_name = '__'.join([f'{key}={value}' for key, value in hyperparameters.items()])
                experiment_name = f'{data_name}{"__" if lrsystem_name and data_name else ""}{lrsystem_name}'

                # This dictionary contains all parameters for this experiment run, prefixed
                # by either 'data.' or 'lrsystem.' to avoid naming conflicts. This is used
                # for reporting purposes in the aggregations.
                parameters = {f'data.{k}': v for k, v in dataparameter.items()} | {
                    f'lrsystem.{k}': v for k, v in hyperparameters.items()
                }

                self._run_lrsystem(lrsystem_config, split_data, parameters, experiment_name, data_config)
