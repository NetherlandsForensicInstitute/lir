from collections.abc import Callable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import optuna

from lir.aggregation import Aggregation
from lir.config.aggregation import parse_aggregations
from lir.config.base import check_is_empty, config_parser, pop_field
from lir.config.lrsystem_architectures import augment_config
from lir.config.metrics import parse_individual_metric
from lir.config.substitution import (
    ContextAwareDict,
    FloatHyperparameter,
    Hyperparameter,
    HyperparameterOption,
    parse_config_with_parameters,
)
from lir.data.models import LLRData
from lir.experiments import Experiment
from lir.experiments.config import pop_experiment_name
from lir.experiments.execution import DataConfig, LRSystemConfig, run_lrsystem


class OptunaExperiment(Experiment):
    """
    An optimization strategy that uses Optuna for choosing parameter values.

    This strategy sequentially runs slight variations of an LR system by changing its hyperparameters. After each run,
    the output is evaluated using a ``metric_function``, and the next set of hyperparameter values is chosen. The
    experiment stops after ``n_trials`` runs are executed.

    Parameters
    ----------
    name : str
        Name used to identify this object in outputs and logs.
    data_config : ContextAwareDict
        Data configuration used to construct datasets for runs.
    outputs : Sequence[Aggregation]
        Output aggregation definitions executed after each run.
    output_path : Path
        Path where generated outputs are written.
    baseline_config : ContextAwareDict
        Baseline configuration to be tuned during optimisation.
    lrsystem_parameters : list[Hyperparameter]
        LR system parameters varied during optimisation.
    n_trials : int
        Number of optimisation trials to execute.
    metric_function : Callable[[LLRData], float]
        Value passed via ``metric_function``.
    """

    def __init__(
        self,
        name: str,
        data_config: ContextAwareDict,
        outputs: Sequence[Aggregation],
        output_path: Path,
        baseline_config: ContextAwareDict,
        lrsystem_parameters: list[Hyperparameter],
        n_trials: int,
        metric_function: Callable[[LLRData], float],
    ):
        super().__init__(name, outputs, output_path)

        self._data_config = DataConfig(spec=data_config, params={}, experiment_output_dir=output_path)

        self.baseline_config = baseline_config
        self.lrsystem_parameters = lrsystem_parameters
        self.n_trials = n_trials
        self.metric_function = metric_function

    @staticmethod
    def _get_parameter_value(trial: optuna.Trial, param: Hyperparameter) -> HyperparameterOption:
        if isinstance(param, FloatHyperparameter):
            value = trial.suggest_float(
                param.path,
                low=param.low,
                high=param.high,
                step=param.step,
                log=param.log,
            )
            return HyperparameterOption(str(value), {param.path: value})
        else:
            options = {option.name: option for option in param.options()}
            selected_option_name = trial.suggest_categorical(param.name, list(options.keys()))
            return options[selected_option_name]

    def _get_hyperparameter_substitutions(self, trial: optuna.Trial) -> dict[str, HyperparameterOption]:
        assignments = {}
        for param in self.lrsystem_parameters:
            assignments[param.name] = self._get_parameter_value(trial, param)

        return assignments

    def _objective(self, trial: optuna.Trial) -> float:
        assignments = self._get_hyperparameter_substitutions(trial)
        lrsystem = augment_config(deepcopy(self.baseline_config), assignments)

        # add optuna values as system parameters
        lrsystem_parameters: dict[str, Any] = assignments
        lrsystem_parameters.update(
            {
                # trial.number is a sequence number, starting at 0
                'trial': trial.number,
                # the best trial is known only at the second run, since the first trial results are not available yet
                'best_trial': trial.study.best_trial.number if trial.number > 0 else '',
            }
        )

        result = run_lrsystem(
            self.output_path,
            LRSystemConfig(spec=lrsystem, params=lrsystem_parameters, experiment_output_dir=self.output_path),
            self._data_config,
            run_name=f'trial{trial.number:03d}',
        )

        for output in self.outputs:
            output.report(result)

        return self.metric_function(result.llrdata)

    def _generate_and_run(self) -> None:
        study = optuna.create_study()  # Create a new study.
        study.optimize(self._objective, n_trials=self.n_trials)  # Invoke optimization of the objective function.


@config_parser
def parse_optuna_experiment(config: ContextAwareDict, output_dir: Path) -> OptunaExperiment:
    """
    Get experiment for an Optuna optimisation strategy.

    Parameters
    ----------
    config : ContextAwareDict
        Experiment configuration section.
    output_dir : Path
        Output directory for the experiment.

    Returns
    -------
    OptunaExperiment
        Optuna-backed experiment.
    """
    name = pop_experiment_name(config)
    experiment_output_dir = output_dir / name

    baseline_config, parameters = parse_config_with_parameters(
        config, experiment_output_dir, 'lrsystem', 'lrsystem_parameters'
    )
    n_trials = pop_field(config, 'n_trials', validate=int)

    metric_name = pop_field(config, 'primary_metric')
    primary_metric = parse_individual_metric(metric_name, experiment_output_dir, config.context)

    data_config = pop_field(config, 'data')

    output_config = pop_field(config, 'output', required=False)
    aggregations = parse_aggregations(output_config, experiment_output_dir) if output_config else []

    check_is_empty(config)

    return OptunaExperiment(
        name=name,
        data_config=data_config,
        outputs=aggregations,
        output_path=experiment_output_dir,
        baseline_config=baseline_config,
        lrsystem_parameters=parameters,
        n_trials=n_trials,
        metric_function=primary_metric,
    )
