from collections import OrderedDict
from collections.abc import Mapping
from functools import partial
from pathlib import Path

from lir import registry
from lir.config.base import (
    check_type,
    pop_field,
)
from lir.config.substitution import (
    ContextAwareDict,
)
from lir.experiments.base_experiment import Experiment


def parse_experiment_strategy(config: ContextAwareDict, output_path: Path) -> Experiment:
    """
    Instantiate the corresponding experiment strategy class, e.g. for a single or grid run.

    A corresponding Experiment class is returned.

    Parameters
    ----------
    config : ContextAwareDict
        Experiment strategy configuration.
    output_path : Path
        Output path for experiment artefacts.

    Returns
    -------
    Experiment
        Parsed experiment strategy instance.
    """
    strategy_name = pop_field(config, 'strategy')
    strategy_parser = registry.get(strategy_name, search_path=['experiment_strategies'])
    return strategy_parser.parse(config, output_path)


def parse_experiments(cfg: ContextAwareDict, output_path: Path) -> Mapping[str, Experiment]:
    """
    Extract which Experiment to run as dictated in the configuration.

    Parameters
    ----------
    cfg : ContextAwareDict
        Configuration section describing experiments.
    output_path : Path
        Filesystem path to the results directory.

    Returns
    -------
    Mapping[str, Experiment]
        Mapping from experiment name to parsed experiment.
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
