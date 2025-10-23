from collections.abc import Callable, Mapping, Sequence
import datetime
from abc import abstractmethod, ABC
from itertools import product
from pathlib import Path

import confidence

from lir import registry
from lir.aggregation import WriteMetricsToCsv, Aggregation
from lir.config.base import (
    YamlParseError,
    check_is_empty,
    pop_field,
    GenericFunctionConfigParser,
    ConfigParser,
    ContextAwareList,
)
from lir.config.data_strategies import parse_data_strategy
from lir.config.lrsystem_architectures import (
    parse_lrsystem,
    parse_augmented_lrsystem,
)
from lir.config.substitution import (
    Hyperparameter,
    parse_hyperparameter,
    _expand,
    ContextAwareDict,
)
from lir.config.visualization import parse_visualizations
from lir.data.models import DataStrategy
from lir.experiment import Experiment, PredefinedExperiment
from lir.optuna import OptunaExperiment
from lir.registry import ComponentNotFoundError


def parse_metric(name: str, output_path: Path, context: list[str]) -> Callable:
    try:
        parser = registry.get(
            name,
            default_config_parser=GenericFunctionConfigParser,
            search_path=["metrics"],
        )
        return parser.parse(ContextAwareDict(context), output_path)
    except ComponentNotFoundError as e:
        raise YamlParseError(context, str(e))


def parse_metrics_section(config: ContextAwareList, output_path: Path) -> Mapping[str, Callable]:
    """Parse the metrics section from the configuration.

    A resulting mapping of metric name and corresponding function is returned.

    Each metric function takes two arguments:
        (1) a numpy array of lrs, and;
        (2) a numpy array of labels.
    The metric function should return the value of the metric.
    The metrics are looked up by their name in the registry.
    """
    return {metric_name: parse_metric(metric_name, output_path, config.context) for metric_name in config}


class ExperimentStrategyConfigParser(ConfigParser, ABC):
    """
    Base class for an experiment strategy configuration parser.
    """

    def __init__(self) -> None:
        self._config: ContextAwareDict
        self._output_dir: Path

    def data(self) -> DataStrategy:
        return parse_data_strategy(pop_field(self._config, "data"), self._output_dir)

    def primary_metric(self) -> Callable:
        metric_name = pop_field(self._config, "primary_metric")
        return parse_metric(metric_name, self._output_dir, self._config.context)

    def aggregations(self) -> Sequence[Aggregation]:
        metrics = parse_metrics_section(pop_field(self._config, "metrics"), self._output_dir)
        return [WriteMetricsToCsv(self._output_dir / "metrics.csv", metrics)]

    def visualization_functions(self) -> list[Callable]:
        return parse_visualizations(pop_field(self._config, "visualization"), self._output_dir)

    def lrsystem(self) -> tuple[ContextAwareDict, list[Hyperparameter]]:
        baseline_config = pop_field(self._config, "lr_system")
        if baseline_config is None:
            baseline_config = ContextAwareDict(self._config.context + ["lr_system"])

        parameters = []
        if "hyperparameters" in self._config.keys():
            parameters = self._config.pop("hyperparameters")
            parameters = [
                parse_hyperparameter(
                    variable,
                    self._output_dir,
                )
                for variable in parameters
            ]

        return baseline_config, parameters

    @abstractmethod
    def get_experiment(self, name: str) -> Experiment:
        raise NotImplementedError

    def parse(
        self,
        config: ContextAwareDict,
        output_dir: Path,
    ) -> Experiment:
        self._config = config
        self._output_dir = output_dir

        exp = self.get_experiment(self._config.context[-1])
        check_is_empty(config)
        return exp


class SingleRunStrategy(ExperimentStrategyConfigParser):
    """Prepare Experiment consisting of a single run using configuration values."""

    def get_experiment(self, name: str) -> Experiment:
        lrsystem = parse_lrsystem(pop_field(self._config, "lr_system"), self._output_dir)

        return PredefinedExperiment(
            name,
            self.data(),
            self.aggregations(),
            self.visualization_functions(),
            self._output_dir,
            [lrsystem],
        )


class GridStrategy(ExperimentStrategyConfigParser):
    """Prepare Excperiment consisting of multiple runs using configuration values."""

    def get_experiment(self, name: str) -> Experiment:
        baseline_config, parameters = self.lrsystem()

        lrsystems = []
        names = [param.name for param in parameters]
        values = [param.options() for param in parameters]
        for value_set in product(*values):
            substitutions = dict(zip(names, value_set))
            lrsystem = parse_augmented_lrsystem(baseline_config, substitutions, self._output_dir)
            lrsystems.append(lrsystem)

        return PredefinedExperiment(
            name,
            self.data(),
            self.aggregations(),
            self.visualization_functions(),
            self._output_dir,
            lrsystems,
        )


class OptunaStrategy(ExperimentStrategyConfigParser):
    """Prepare Excperiment for optimizing configuration parameters."""

    def get_experiment(self, name: str) -> Experiment:
        baseline_config, parameters = self.lrsystem()
        n_trials = pop_field(self._config, "n_trials", validate=int)
        return OptunaExperiment(
            name,
            self.data(),
            self.aggregations(),
            self.visualization_functions(),
            self._output_dir,
            baseline_config,
            parameters,
            n_trials,
            self.primary_metric(),
        )


def parse_experiment_strategy(config: ContextAwareDict, output_path: Path) -> Experiment:
    """Instantiate the corresponding experiment strategy class, e.g. for a single or grid run.

    A corresponding Experiment class is returned.
    """
    strategy_name = pop_field(config, "strategy")
    if strategy_name is None:
        raise YamlParseError(config.context, "Missing strategy name.")
    strategy_parser = registry.get(strategy_name, search_path=["experiment_strategies"])
    return strategy_parser.parse(config, output_path / config.context[-1])


def parse_experiments(cfg: ContextAwareDict, output_path: Path) -> Mapping[str, Experiment]:
    """
    Extract which Experiment to run as dictated in the configuration.

    :param cfg: a `dict` object describing the experiments
    :param output_path: the filesystem path to the results directory
    :return: a mapping of names to experiments
    """
    experiments: dict[str, Experiment] = {}
    experiments_config_section = pop_field(cfg, "experiments")
    if not experiments_config_section:
        return experiments

    if not isinstance(experiments_config_section, Mapping):
        raise YamlParseError(cfg.context, "invalid value for experiments")

    for exp_name, exp_config in experiments_config_section.items():
        experiment = parse_experiment_strategy(
            exp_config,
            output_path,
        )
        experiments[exp_name] = experiment

    return experiments


def parse_experiments_setup(
    cfg: confidence.Configuration,
) -> tuple[Mapping[str, Experiment], Path]:
    """
    Extract which Experiment to run as dictated in the configuration.

    The following pre-defined variables are injected to the configuration:

    - `timestamp`: a formatted timestamp of the current date/time

    :param cfg: a `Configuration` object describing the experiments
    :return: a tuple with two elements: (1) mapping of names to experiments; (2) path to output directory
    """
    cfg = confidence.Configuration(cfg, {"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")})

    cfg = _expand([], cfg)

    output_dir = pop_field(cfg, "output", validate=Path)
    return parse_experiments(cfg, output_dir), output_dir
