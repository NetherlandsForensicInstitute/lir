from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from tqdm import tqdm

import lir
from lir.aggregation import Aggregation
from lir.config.base import ContextAwareDict
from lir.config.data import parse_data_object
from lir.experiments import Experiment


class PredefinedExperiment(Experiment):
    """Representation of an experiment run for each provided LR system."""

    def __init__(
        self,
        name: str,
        data_configs: list[tuple[ContextAwareDict, dict[str, Any]]],
        outputs: Sequence[Aggregation],
        output_path: Path,
        lrsystem_configs: list[tuple[ContextAwareDict, dict[str, Any]]],
    ):
        super().__init__(name, outputs, output_path)
        self.lrsystem_configs = lrsystem_configs
        self.data_configs = data_configs

    def _generate_and_run(self) -> None:
        # Only display the data configuration progress bar when running interactively and
        # there are multiple data configurations to evaluate.
        disable_data_tqdm = not lir.is_interactive() or len(self.data_configs) == 1

        for data_config, data_parameter in tqdm(self.data_configs, disable=disable_data_tqdm, desc=self.name):
            # Parse the data configuration. This is done here to ensure that data
            # parsing is only done once per data configuration, even when multiple
            # LR systems are being evaluated on the same data setup.
            provider, splitter = parse_data_object(deepcopy(data_config), self.output_path)
            split_data = list(splitter.apply(provider.get_instances()))

            # Only display the LR system configuration progress bar when running interactively, and
            # the data_tqdm is not disabled, and there are multiple LR systems to evaluate.
            disable_lrsystem_tqdm = not lir.is_interactive() or not disable_data_tqdm or len(self.lrsystem_configs) == 1
            for lrsystem_config, lrsystem_parameters in tqdm(
                self.lrsystem_configs, desc=self.name, disable=disable_lrsystem_tqdm
            ):
                # Combine the data parameter with the LR system parameters to create a unique name for this run.
                data_name = '__'.join([f'{key}={value}' for key, value in data_parameter.items()])
                lrsystem_name = '__'.join([f'{key}={value}' for key, value in lrsystem_parameters.items()])
                run_name = f'{data_name}{"__" if lrsystem_name and data_name else ""}{lrsystem_name}'

                # This dictionary contains all parameters for this experiment run, prefixed
                # by either 'data.' or 'lrsystem.' to avoid naming conflicts. This is used
                # for reporting purposes in the aggregations.
                parameters = {f'data.{k}': v for k, v in data_parameter.items()} | {
                    f'lrsystem.{k}': v for k, v in lrsystem_parameters.items()
                }

                self._run_lrsystem(lrsystem_config, split_data, parameters, run_name, data_config)
