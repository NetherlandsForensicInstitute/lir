import datetime
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from itertools import product
from pathlib import Path

import confidence

from lir import registry
from lir.aggregation import Aggregation, WriteMetricsToCsv
from lir.config.base import (
    ConfigParser,
    GenericFunctionConfigParser,
    YamlParseError,
    check_is_empty,
    config_parser,
    pop_field,
)
from lir.config.data_strategies import parse_data_strategy
from lir.config.lrsystem_architectures import (
    parse_augmented_lrsystem,
    parse_lrsystem,
)
from lir.config.substitution import (
    ContextAwareDict,
    Hyperparameter,
    _expand,
    parse_hyperparameter,
)
from lir.data.models import DataStrategy
from lir.experiment import Experiment, PredefinedExperiment
from lir.optuna import OptunaExperiment
from lir.registry import ComponentNotFoundError


def parse_metric(name: str, output_path: Path, context: list[str]) -> Callable:
    try:
        parser = registry.get(
            name,
            default_config_parser=GenericFunctionConfigParser,
            search_path=['metrics'],
        )
        return parser.parse(ContextAwareDict(context), output_path)
    except ComponentNotFoundError as e:
        raise YamlParseError(context, str(e))


@config_parser
def metrics_csv(config: ContextAwareDict, output_dir: Path) -> WriteMetricsToCsv:
    metrics = pop_field(config, 'columns', required=True)
    if not isinstance(metrics, Sequence):
        raise YamlParseError(
            config.context,
            'Invalid metrics configuration; expected a list of metric names.',
        )

    metrics = {name: parse_metric(name, output_dir, config.context) for name in metrics}
    return WriteMetricsToCsv(output_dir, metrics)


class ExperimentStrategyConfigParser(ConfigParser, ABC):
    """
    Base class for an experiment strategy configuration parser.
    """

    def __init__(self) -> None:
        self._config: ContextAwareDict
        self._output_dir: Path

    def data(self) -> DataStrategy:
        return parse_data_strategy(pop_field(self._config, 'data'), self._output_dir)

    def primary_metric(self) -> Callable:
        metric_name = pop_field(self._config, 'primary_metric')
        return parse_metric(metric_name, self._output_dir, self._config.context)

    def output_list(self) -> Sequence[Aggregation]:
        config: ContextAwareDict = pop_field(self._config, 'output', required=False)
        if not config:
            return []

        results: list[Aggregation] = []
        for item in config:
            # Normalise configuration into (class_name, args)
            if isinstance(item, str):
                class_name, args = item, ContextAwareDict(config.context)
            elif isinstance(item, Mapping):
                class_name = pop_field(item, 'method')
                args = ContextAwareDict(config.context, item)
            else:
                raise YamlParseError(
                    config.context,
                    'Invalid output configuration; expected a string or a mapping with a "method" field.',
                )

            args['output_dir'] = self._output_dir

            parser: ConfigParser = registry.get(
                class_name,
                default_config_parser=GenericFunctionConfigParser,
                search_path=['output'],
            )
            parsed_object = parser.parse(args, self._output_dir)

            # Instantiate if needed (not already done by the parser)
            if isinstance(parsed_object, type):
                parsed_object = parsed_object(**args)

            if not isinstance(parsed_object, Aggregation):
                raise YamlParseError(
                    config.context,
                    f'Invalid output configuration; expected an Aggregation, found: {type(parsed_object)}.',
                )

            results.append(parsed_object)  # type: ignore

        return results

    def lrsystem(self) -> tuple[ContextAwareDict, list[Hyperparameter]]:
        baseline_config = pop_field(self._config, 'lr_system')
        if baseline_config is None:
            baseline_config = ContextAwareDict(self._config.context + ['lr_system'])

        parameters = []
        if 'hyperparameters' in self._config:
            parameters = self._config.pop('hyperparameters')
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
        lrsystem = parse_lrsystem(pop_field(self._config, 'lr_system'), self._output_dir)

        return PredefinedExperiment(
            name,
            self.data(),
            self.output_list(),
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
            substitutions = dict(zip(names, value_set, strict=True))
            lrsystem = parse_augmented_lrsystem(baseline_config, substitutions, self._output_dir)
            lrsystems.append(lrsystem)

        return PredefinedExperiment(
            name,
            self.data(),
            self.output_list(),
            self._output_dir,
            lrsystems,
        )


class OptunaStrategy(ExperimentStrategyConfigParser):
    """Prepare Excperiment for optimizing configuration parameters."""

    def get_experiment(self, name: str) -> Experiment:
        baseline_config, parameters = self.lrsystem()
        n_trials = pop_field(self._config, 'n_trials', validate=int)
        return OptunaExperiment(
            name,
            self.data(),
            self.output_list(),
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
    strategy_name = pop_field(config, 'strategy')
    if strategy_name is None:
        raise YamlParseError(config.context, 'Missing strategy name.')
    strategy_parser = registry.get(strategy_name, search_path=['experiment_strategies'])
    return strategy_parser.parse(config, output_path / config.context[-1])


def parse_experiments(cfg: ContextAwareDict, output_path: Path) -> Mapping[str, Experiment]:
    """
    Extract which Experiment to run as dictated in the configuration.

    :param cfg: a `dict` object describing the experiments
    :param output_path: the filesystem path to the results directory
    :return: a mapping of names to experiments
    """
    experiments: dict[str, Experiment] = {}
    experiments_config_section = pop_field(cfg, 'experiments')
    if not experiments_config_section:
        return experiments

    if not isinstance(experiments_config_section, Mapping):
        raise YamlParseError(cfg.context, 'invalid value for experiments')

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
    cfg = confidence.Configuration(cfg, {'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')})  # noqa: DTZ005

    cfg = _expand([], cfg)

    output_dir = pop_field(cfg, 'output', validate=Path)
    return parse_experiments(cfg, output_dir), output_dir
