from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from itertools import product
from pathlib import Path

from lir import registry
from lir.aggregation import Aggregation
from lir.config.base import (
    ConfigParser,
    GenericConfigParser,
    YamlParseError,
    check_is_empty,
    check_type,
    pop_field,
)
from lir.config.data_providers import parse_data_provider
from lir.config.data_strategies import parse_data_strategy
from lir.config.lrsystem_architectures import (
    parse_augmented_lrsystem,
    parse_lrsystem,
)
from lir.config.metrics import parse_individual_metric
from lir.config.substitution import (
    ContextAwareDict,
    Hyperparameter,
    parse_hyperparameter,
)
from lir.data.models import DataProvider, DataStrategy
from lir.experiment import Experiment, PredefinedExperiment
from lir.optuna import OptunaExperiment


class ExperimentStrategyConfigParser(ConfigParser, ABC):
    """Base class for an experiment strategy configuration parser."""

    def __init__(self) -> None:
        self._config: ContextAwareDict
        self._output_dir: Path

    def data(self) -> tuple[DataProvider, DataStrategy]:
        """Parse the data section of the configuration.

        The corresponding data provider and splitting strategy instances are provided.
        """
        data_section = pop_field(self._config, 'data')
        provider = parse_data_provider(pop_field(data_section, 'provider'), self._output_dir)
        splitter = parse_data_strategy(pop_field(data_section, 'splits'), self._output_dir)
        check_is_empty(data_section)
        return provider, splitter

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

        results: list[Aggregation] = []
        for i, item in enumerate(config):
            # Normalise configuration into (class_name, args)
            if isinstance(item, str):
                class_name, args = item, ContextAwareDict(config.context + [str(i)])
            elif isinstance(item, ContextAwareDict):
                class_name = pop_field(item, 'method')
                args = item
            else:
                raise YamlParseError(
                    config.context,
                    'Invalid output configuration; expected a string or a mapping with a "method" field.',
                )

            parser: ConfigParser = registry.get(
                class_name,
                default_config_parser=GenericConfigParser,
                search_path=['output'],
            )
            parsed_object = parser.parse(args, self._output_dir)

            if not isinstance(parsed_object, Aggregation):
                raise YamlParseError(
                    config.context,
                    f'Invalid output configuration; expected an Aggregation, found: {type(parsed_object)}.',
                )

            results.append(parsed_object)  # type: ignore

        return results

    def lrsystem(self) -> tuple[ContextAwareDict, list[Hyperparameter]]:
        """Parse the LR System section including hyperparameters.

        The baseline configuration is provided along with the specified parameters to vary (the
        defined hyperparameters).
        """
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
        """Get the experiment by `name` for the defined LR system."""
        raise NotImplementedError

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
        lrsystem = parse_lrsystem(pop_field(self._config, 'lr_system'), self._output_dir)
        data_provider, data_splitter = self.data()

        return PredefinedExperiment(
            name,
            data_provider,
            data_splitter,
            self.output_list(),
            self._output_dir,
            [(lrsystem, {})],
        )


class GridStrategy(ExperimentStrategyConfigParser):
    """Prepare Experiment consisting of multiple runs using configuration values."""

    def get_experiment(self, name: str) -> Experiment:
        """Get experiment for the grid strategy run, based on its name."""
        baseline_config, parameters = self.lrsystem()

        lrsystems = []
        names = [param.name for param in parameters]
        values = [param.options() for param in parameters]
        for value_set in product(*values):
            substitutions = dict(zip(names, value_set, strict=True))
            lrsystem = parse_augmented_lrsystem(baseline_config, substitutions, self._output_dir)
            lrsystems.append((lrsystem, substitutions))

        data_provider, data_splitter = self.data()

        return PredefinedExperiment(
            name,
            data_provider,
            data_splitter,
            self.output_list(),
            self._output_dir,
            lrsystems,
        )


class OptunaStrategy(ExperimentStrategyConfigParser):
    """Prepare Experiment for optimizing configuration parameters."""

    def get_experiment(self, name: str) -> Experiment:
        """Get experiment for the optuna run, based on its name."""
        baseline_config, parameters = self.lrsystem()
        n_trials = pop_field(self._config, 'n_trials', validate=int)
        data_provider, data_splitter = self.data()

        return OptunaExperiment(
            name,
            data_provider,
            data_splitter,
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
