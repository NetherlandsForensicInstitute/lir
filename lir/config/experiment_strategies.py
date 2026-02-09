from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any

from lir import registry
from lir.aggregation import Aggregation
from lir.config.aggregation import parse_aggregations
from lir.config.base import (
    ConfigParser,
    check_is_empty,
    check_type,
    pop_field,
)
from lir.config.lrsystem_architectures import (
    augment_config,
)
from lir.config.metrics import parse_individual_metric
from lir.config.substitution import (
    ContextAwareDict,
    Hyperparameter,
    parse_parameter,
)
from lir.experiment import Experiment, PredefinedExperiment
from lir.optuna import OptunaExperiment


class ExperimentStrategyConfigParser(ConfigParser, ABC):
    """Base class for an experiment strategy configuration parser."""

    def __init__(self) -> None:
        self._config: ContextAwareDict
        self._output_dir: Path

    def primary_metric(self) -> Callable:
        """Parse the `primary_metric` field."""
        metric_name = pop_field(self._config, 'primary_metric')
        return parse_individual_metric(metric_name, self._output_dir, self._config.context)

    def output_list(self) -> Sequence[Aggregation]:
        """Initialize corresponding aggregation classes based on the `output` section.

        The initialized aggregation classes are returned as a sequence, to be iterated over in a
        later stage.
        """
        config: ContextAwareDict = pop_field(self._config, 'output', required=False)
        if not config:
            return []

        return parse_aggregations(config, self._output_dir)

    @abstractmethod
    def get_experiment(self, name: str) -> Experiment:
        """Get the experiment by `name` for the defined LR system."""
        raise NotImplementedError

    def _parse_config_with_parameters(
        self,
        config_field: str,
        parameters_field: str,
    ) -> tuple[ContextAwareDict, list[Hyperparameter]]:
        """Extract a configuration section and its associated parameters.

        :param config_field: the name of the field containing the baseline configuration
        :param parameters_field: the name of the field containing the parameters to vary
        :return: a tuple of (baseline_config, list of hyperparameters)
        """
        baseline_config = pop_field(self._config, config_field)
        if baseline_config is None:
            baseline_config = ContextAwareDict(self._config.context + [config_field])

        parameters = []
        if parameters_field in self._config:
            parameters = self._config.pop(parameters_field)
            parameters = [parse_parameter(variable, self._output_dir) for variable in parameters]

        return baseline_config, parameters

    def data_config(self) -> tuple[ContextAwareDict, list[Hyperparameter]]:
        """Prepare the data provider and data strategy from the configuration.

        The (hyper)parameters to vary for the data provider and data strategy are also parsed.
        """
        return self._parse_config_with_parameters('data', 'dataparameters')

    def lrsystem_config(self) -> tuple[ContextAwareDict, list[Hyperparameter]]:
        """Parse the LR System section including hyperparameters.

        The baseline configuration is provided along with the specified parameters to vary (the
        defined hyperparameters).
        """
        return self._parse_config_with_parameters('lr_system', 'hyperparameters')

    def parse(
        self,
        config: ContextAwareDict,
        output_dir: Path,
    ) -> Experiment:
        """Parse the experiment section of the configuration."""
        self._config = config

        experiment_name = pop_field(config, 'name', default=f'unnamed_experiment{config.context[-1]}')

        self._output_dir = output_dir / experiment_name
        exp = self.get_experiment(experiment_name)

        check_is_empty(config)
        return exp


class SingleRunStrategy(ExperimentStrategyConfigParser):
    """Prepare Experiment consisting of a single run using configuration values."""

    def get_experiment(self, name: str) -> Experiment:
        """Get an experiment for a single run, based on its name."""
        return PredefinedExperiment(
            name,
            [(pop_field(self._config, 'data'), {})],
            self.output_list(),
            self._output_dir,
            [(pop_field(self._config, 'lr_system'), {})],
        )


def create_configs_from_hyperparameters(
    baseline_config: ContextAwareDict,
    parameters: list[Hyperparameter],
) -> list[tuple[ContextAwareDict, dict[str, Any]]]:
    """Create configurations for all combinations of hyperparameter options.

    Generates a Cartesian product of all hyperparameter options and creates a configuration
    for each combination by substituting the values into the baseline configuration.

    This is used for both dataparameters and lrsystem hyperparameters in grid search.

    :param baseline_config: the baseline configuration to augment
    :param parameters: the hyperparameters to vary
    :return: a list of tuples, where each tuple contains:
        - the augmented configuration with substituted values
        - a dict mapping parameter names to the substituted values
    """
    configs = []
    parameter_names = [param.name for param in parameters]
    parameter_values = [param.options() for param in parameters]

    for value_set in product(*parameter_values):
        substitutions = dict(zip(parameter_names, value_set, strict=True))
        substituted_config = augment_config(baseline_config, substitutions)
        configs.append((substituted_config, substitutions))

    return configs


class GridStrategy(ExperimentStrategyConfigParser):
    """Prepare Experiment consisting of multiple runs using configuration values."""

    def get_experiment(self, name: str) -> Experiment:
        """Get experiment for the grid strategy run, based on its name."""
        lrsystem_configs = create_configs_from_hyperparameters(*self.lrsystem_config())
        data_configs = create_configs_from_hyperparameters(*self.data_config())

        return PredefinedExperiment(
            name,
            data_configs,
            self.output_list(),
            self._output_dir,
            lrsystem_configs,
        )


class OptunaStrategy(ExperimentStrategyConfigParser):
    """Prepare Experiment for optimizing configuration parameters."""

    def get_experiment(self, name: str) -> Experiment:
        """Get experiment for the optuna run, based on its name."""
        baseline_config, parameters = self.lrsystem_config()
        n_trials = pop_field(self._config, 'n_trials', validate=int)

        return OptunaExperiment(
            name=name,
            data_config=self.data_config()[0],
            outputs=self.output_list(),
            output_path=self._output_dir,
            baseline_config=baseline_config,
            hyperparameters=parameters,
            n_trials=n_trials,
            metric_function=self.primary_metric(),
        )


def parse_experiment_strategy(config: ContextAwareDict, output_path: Path) -> Experiment:
    """Instantiate the corresponding experiment strategy class, e.g. for a single or grid run.

    A corresponding Experiment class is returned.
    """
    strategy_name = pop_field(config, 'strategy')
    strategy_parser = registry.get(strategy_name, search_path=['experiment_strategies'])
    return strategy_parser.parse(config, output_path)


def parse_experiments(cfg: ContextAwareDict, output_path: Path) -> Mapping[str, Experiment]:
    """
    Extract which Experiment to run as dictated in the configuration.

    :param cfg: a `dict` object describing the experiments
    :param output_path: the filesystem path to the results directory
    :return: a mapping of names to experiments
    """
    experiments_config_section = pop_field(cfg, 'experiments', validate=partial(check_type, list))

    experiments: OrderedDict[str, Experiment] = OrderedDict()
    for exp_config in experiments_config_section:
        experiment = parse_experiment_strategy(
            exp_config,
            output_path,
        )

        if experiment.name in experiments:
            raise ValueError(f'experiment {experiment.name} already exists')

        experiments[experiment.name] = experiment

    return experiments
