from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import optuna

from lir.aggregation import Aggregation
from lir.config.data import parse_data_object
from lir.config.lrsystem_architectures import parse_augmented_config, parse_lrsystem
from lir.config.substitution import (
    ContextAwareDict,
    FloatHyperparameter,
    Hyperparameter,
    HyperparameterOption,
)
from lir.data.models import LLRData
from lir.experiment import Experiment


class OptunaExperiment(Experiment):
    """Representation of an experiment run for each provided LR system."""

    def __init__(
        self,
        name: str,
        data_config: ContextAwareDict,
        outputs: Sequence[Aggregation],
        output_path: Path,
        baseline_config: ContextAwareDict,
        hyperparameters: list[Hyperparameter],
        n_trials: int,
        metric_function: Callable[[LLRData], float],
    ):
        super().__init__(name, outputs, output_path)

        self.data_config = data_config
        data_provider, splitter = parse_data_object(data_config, output_path)
        self.split_data = splitter.apply(data_provider.get_instances())

        self.baseline_config = baseline_config
        self.hyperparameters = hyperparameters
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
        for param in self.hyperparameters:
            assignments[param.name] = self._get_parameter_value(trial, param)

        return assignments

    def _objective(self, trial: optuna.Trial) -> float:
        assignments = self._get_hyperparameter_substitutions(trial)
        lrsystem = parse_lrsystem(
            parse_augmented_config(
                self.baseline_config,
                assignments,
            ),
            self.output_path,
        )

        # add optuna values as system parameters
        hyperparameters: dict[str, Any] = assignments
        hyperparameters.update(
            {
                # trial.number is a sequence number, starting at 0
                'trial': trial.number,
                # the best trial is known only at the second run, since the first trial results are not available yet
                'best_trial': trial.study.best_trial.number if trial.number > 0 else '',
            }
        )

        llr_data: LLRData = self._run_lrsystem(lrsystem, self.split_data, str(hyperparameters), self.data_config)

        return self.metric_function(llr_data)

    def _generate_and_run(self) -> None:
        study = optuna.create_study()  # Create a new study.
        study.optimize(self._objective, n_trials=self.n_trials)  # Invoke optimization of the objective function.
