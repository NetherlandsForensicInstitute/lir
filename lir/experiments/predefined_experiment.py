from collections.abc import Iterator, Sequence
from itertools import product
from pathlib import Path
from typing import Any

from tqdm import tqdm

import lir
from lir.aggregation import Aggregation
from lir.config.aggregation import parse_aggregations
from lir.config.base import ContextAwareDict, check_is_empty, config_parser, pop_field
from lir.config.lrsystem_architectures import augment_config
from lir.config.substitution import parse_config_with_parameters
from lir.experiments import Experiment

from .config import pop_experiment_name
from .execution import DataConfig, LRSystemConfig, parallellize_runs, run_multiple


class PredefinedExperiment(Experiment):
    """
    Experiment strategy that runs a pre-defined set of LR systems on a pre-defined set of data setups.

    To set up a single run experiment in a YAML configuration:

    .. code-block:: yaml

        experiments:
          - strategy: single_run
            name: my experiment
            data: *my_data_setup
            lr_system: *my_lrsystem
            output: *my_aggregations

    Multiple runs can be defined using the ``grid`` strategy, with additional configuration options:

    - use the ``hyperparameters`` field to configure which hyperparameters can be varied;
    - use the ``dataparameters`` field to configure which dataparameters can be varied;
    - set the ``enable_parallelization`` field to ``True`` to enable parallelization.

    For more guidance and working examples, see: :ref:`experiment-setup`.

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
        data_configs: list[DataConfig],
        outputs: Sequence[Aggregation],
        output_path: Path,
        lrsystem_configs: list[LRSystemConfig],
        enable_parallelization: bool = False,
    ):
        super().__init__(name, outputs, output_path)
        self._lrsystem_configs = lrsystem_configs
        self._data_configs = data_configs
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


@config_parser
def parse_single_run(config: ContextAwareDict, output_dir: Path) -> PredefinedExperiment:
    """
    Get an experiment for a single run.

    Parameters
    ----------
    config : ContextAwareDict
        Experiment strategy configuration.
    output_dir : Path
        Base output directory for this experiment.

    Returns
    -------
    PredefinedExperiment
        Predefined single-run experiment.
    """
    name = pop_experiment_name(config)

    output_config = pop_field(config, 'output', required=False)
    aggregations = parse_aggregations(output_config, output_dir) if output_config else []

    exp = PredefinedExperiment(
        name,
        [DataConfig(pop_field(config, 'data'), {}, output_dir)],
        aggregations,
        output_dir / name,
        [LRSystemConfig(pop_field(config, 'lr_system'), {}, output_dir)],
    )
    check_is_empty(config)
    return exp


@config_parser
def parse_grid_experiment(config: ContextAwareDict, output_dir: Path) -> PredefinedExperiment:
    """
    Get experiment for a grid search strategy.

    Parameters
    ----------
    config : ContextAwareDict
        Experiment strategy configuration.
    output_dir : Path
        Base output directory for this experiment.

    Returns
    -------
    PredefinedExperiment
        Predefined experiment with all parameter combinations.
    """
    name = pop_experiment_name(config)

    output_config = pop_field(config, 'output', required=False)
    aggregations = parse_aggregations(output_config, output_dir) if output_config else []

    lrsystem_configs = [
        LRSystemConfig(*cfg, output_dir)
        for cfg in _create_configs_from_hyperparameters(config, output_dir, 'lr_system', 'hyperparameters')
    ]
    data_configs = [
        DataConfig(*cfg, output_dir)
        for cfg in _create_configs_from_hyperparameters(config, output_dir, 'data', 'dataparameters')
    ]
    enable_parallelization = pop_field(config, 'enable_parallelization', validate=bool, default=False)
    check_is_empty(config)

    return PredefinedExperiment(
        name,
        data_configs,
        aggregations,
        output_dir / name,
        lrsystem_configs,
        enable_parallelization=enable_parallelization,
    )


def _create_configs_from_hyperparameters(
    config: ContextAwareDict,
    output_dir: Path,
    baseline_field: str,
    parameters_field: str,
) -> Iterator[tuple[ContextAwareDict, dict[str, Any]]]:
    """
    Create configurations for all combinations of hyperparameter options.

    Generates a Cartesian product of all hyperparameter options and creates a configuration
    for each combination by substituting the values into the baseline configuration.

    This is used for both dataparameters and lrsystem hyperparameters in grid search.

    Parameters
    ----------
    config : ContextAwareDict
        The data configuration section.
    output_dir : Path
        Output directory path.
    baseline_field : str
        The field name of the baseline configuration within the data configuration section.
    parameters_field : str
        The field name of the data parameters within the data configuration section.

    Returns
    -------
    list[tuple[ContextAwareDict, dict[str, Any]]]
        Augmented configurations and applied substitutions.
    """
    baseline_config, parameters = parse_config_with_parameters(config, output_dir, baseline_field, parameters_field)

    parameter_names = [param.name for param in parameters]
    parameter_values = [param.options() for param in parameters]

    for value_set in product(*parameter_values):
        substitutions = dict(zip(parameter_names, value_set, strict=True))
        substituted_config = augment_config(baseline_config, substitutions)

        yield substituted_config, substitutions
