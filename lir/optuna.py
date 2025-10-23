from collections.abc import Callable, Sequence
from pathlib import Path

import optuna

from lir import compat
from lir.aggregation import Aggregation
from lir.config.lrsystem_architectures import parse_augmented_lrsystem
from lir.config.substitution import (
    Hyperparameter,
    FloatHyperparameter,
    HyperparameterOption,
    ContextAwareDict,
)
from lir.data.models import DataStrategy
from lir.experiment import Experiment


class OptunaExperiment(Experiment):
    """Representation of an experiment run for each provided LR system."""

    def __init__(
        self,
        name: str,
        data: DataStrategy,
        aggregations: Sequence[Aggregation],
        visualization_functions: list[Callable],
        output_path: Path,
        baseline_config: ContextAwareDict,
        hyperparameters: list[Hyperparameter],
        n_trials: int,
        metric_function: Callable,
    ):
        super().__init__(name, data, aggregations, visualization_functions, output_path)
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

    def _get_hypparameter_substitutions(self, trial: optuna.Trial) -> dict[str, HyperparameterOption]:
        assignments = {}
        for param in self.hyperparameters:
            assignments[param.name] = self._get_parameter_value(trial, param)

        return assignments

    def _objective(self, trial: optuna.Trial) -> float:
        assignments = self._get_hypparameter_substitutions(trial)
        lrsystem = parse_augmented_lrsystem(
            self.baseline_config,
            assignments,
            self.output_path,
            dirname_prefix=f"{trial.number:03d}__",
        )

        # add optuna values as system parameters
        lrsystem.parameters.update(
            {
                # trial.number is a sequence number, starting at 0
                "trial": trial.number,
                # the best trial is known only at the second run, since the first trial results are not available yet
                "best_trial": trial.study.best_trial.number if trial.number > 0 else "",
            }
        )

        llrs, labels = self._run_lrsystem(lrsystem)
        return self.metric_function(compat.llr_to_lr(llrs), labels)

    def _generate_and_run(self) -> None:
        study = optuna.create_study()  # Create a new study.
        study.optimize(self._objective, n_trials=self.n_trials)  # Invoke optimization of the objective function.
