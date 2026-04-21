from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tqdm import tqdm

import lir
from lir.aggregation import Aggregation
from lir.config.base import ContextAwareDict
from lir.config.execution import DataConfig, LRSystemConfig, parallellize_runs, run_multiple
from lir.experiments import Experiment


class PredefinedExperiment(Experiment):
    """
    Representation of an experiment run for each provided LR system.

    Parameters
    ----------
    name : str
        Name used to identify this object in outputs and logs.
    data_configs : list[tuple[ContextAwareDict, dict[str, Any]]]
        Data configurations evaluated by this experiment.
    outputs : Sequence[Aggregation]
        Output aggregation definitions executed after each run.
    output_path : Path
        Path where generated outputs are written.
    lrsystem_configs : list[tuple[ContextAwareDict, dict[str, Any]]]
        LR-system configurations evaluated by this experiment.
    enable_parallelization : bool
        Whether to run the LR systems in parallel.
    """

    def __init__(
        self,
        name: str,
        data_configs: list[tuple[ContextAwareDict, dict[str, Any]]],
        outputs: Sequence[Aggregation],
        output_path: Path,
        lrsystem_configs: list[tuple[ContextAwareDict, dict[str, Any]]],
        enable_parallelization: bool = False,
    ):
        super().__init__(name, outputs, output_path)
        self._lrsystem_configs = [
            LRSystemConfig(spec=cfg, params=params, output_dir=output_path) for cfg, params in lrsystem_configs
        ]
        self._data_configs = [
            DataConfig(spec=cfg, params=params, output_dir=output_path) for cfg, params in data_configs
        ]
        self._enable_parallelization = enable_parallelization

    def _generate_and_run(self) -> None:
        # Only display the progress bar when running interactively and there are multiple configurations to evaluate.
        number_of_runs = len(self._lrsystem_configs) * len(self._data_configs)
        disable_tqdm = not lir.is_interactive() or number_of_runs == 1

        progress = tqdm(desc=self.name, total=number_of_runs, disable=disable_tqdm)
        run_func = parallellize_runs if self._enable_parallelization else run_multiple
        for result in run_func(self.output_path, self._lrsystem_configs, self._data_configs):
            for output in self.outputs:
                output.report(result)
            progress.update(1)
        progress.close()
