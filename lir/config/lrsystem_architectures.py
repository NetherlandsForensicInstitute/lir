import itertools
import logging
from pathlib import Path

from lir import registry
from lir.config.base import (
    YamlParseError,
    config_parser,
    parse_pairing_config,
    pop_field,
)
from lir.config.substitution import (
    ContextAwareDict,
    HyperparameterOption,
    substitute_hyperparameters,
)
from lir.config.transform import parse_module
from lir.lrsystems.binary_lrsystem import BinaryLRSystem
from lir.lrsystems.lrsystems import LRSystem
from lir.lrsystems.score_based import ScoreBasedSystem
from lir.lrsystems.two_level import TwoLevelSystem
from lir.registry import ComponentNotFoundError
from lir.transform.pipeline import Pipeline


LOG = logging.getLogger(__name__)


def parse_pipeline(modules_config: ContextAwareDict, output_dir: Path) -> Pipeline:
    """Construct a scikit-learn Pipeline based on the provided configuration."""
    if modules_config is None:
        return Pipeline([])

    module_names = list(modules_config.keys())
    modules = [
        (
            module_name,
            parse_module(
                pop_field(modules_config, module_name),
                output_dir,
                modules_config.context + [module_name],
            ),
        )
        for module_name in module_names
    ]

    return Pipeline(modules)


@config_parser
def specific_source(config: ContextAwareDict, output_dir: Path) -> LRSystem:
    """Construct a specific-source LR system based on the provided configuration.

    The `specific_source` function name corresponds with the naming scheme in the
    registry. See for example: `lir.config.lrsystems.specific_source`.
    """
    pipeline = parse_pipeline(pop_field(config, 'modules'), output_dir)
    return BinaryLRSystem(output_dir.name, pipeline)


@config_parser
def score_based(config: ContextAwareDict, output_dir: Path) -> LRSystem:
    """Construct a specific-source LR system based on the provided configuration.

    The `specific_source` function name corresponds with the naming scheme in the
    registry. See for example: `lir.config.lrsystems.score_based`.
    """
    preprocessing = parse_pipeline(pop_field(config, 'preprocessing'), output_dir)
    pairing_function = parse_pairing_config(pop_field(config, 'pairing'), output_dir, config.context)
    evaluation = parse_pipeline(pop_field(config, 'comparing'), output_dir)

    return ScoreBasedSystem(output_dir.name, preprocessing, pairing_function, evaluation)


@config_parser
def two_level(config: ContextAwareDict, output_dir: Path) -> LRSystem:
    preprocessing = parse_pipeline(pop_field(config, 'preprocessing'), output_dir)
    pairing_function = parse_pairing_config(pop_field(config, 'pairing'), output_dir, config.context)
    postprocessing = parse_pipeline(pop_field(config, 'postprocessing'), output_dir)
    n_trace_instances = pop_field(config, 'n_trace_instances', validate=int)
    n_ref_instances = pop_field(config, 'n_ref_instances', validate=int)

    return TwoLevelSystem(
        output_dir.name,
        preprocessing,
        pairing_function,
        postprocessing,
        n_trace_instances,
        n_ref_instances,
    )


def parse_lrsystem(config: ContextAwareDict, output_dir: Path) -> LRSystem:
    """Determine and initialise corresponding LR system from configuration values.

    LR systems are provided under the `architectures` key.
    """
    architecture = pop_field(config, 'architecture')

    try:
        parser = registry.get(architecture, search_path=['lrsystem_architectures'])
    except ComponentNotFoundError as e:
        raise YamlParseError(config.context, f'{e}')

    return parser.parse(config, output_dir)


def parse_augmented_lrsystem(
    baseline_lrsystem_config: ContextAwareDict,
    hyperparameters: dict[str, HyperparameterOption],
    output_dir: Path,
    dirname_prefix: str = '',
) -> LRSystem:
    """
    Parses an augmented LR system.

    The LR system is parsed from a base configuration and a set of parameter substitutions that override parts of the
    base configuration. Results are written to a subdirectory of `output_dir` that is named by its parameter
    substitutions and prefixed by `dirname_prefix`.

    :param baseline_lrsystem_config: the base LR system configuration
    :param hyperparameters: hyperparameter substitutions that override parts of the base configuration
    :param context: the configuration context path as a list of str
    :param output_dir: the directory where create a results directory
    :param dirname_prefix: the prefix of the created directory name
    :return:
    """
    substitutions = dict(itertools.chain(*[opt.substitutions.items() for opt in hyperparameters.values()]))

    name = '__'.join([f'{key}={value}' for key, value in hyperparameters.items()])
    context = baseline_lrsystem_config.context + [f'substitution[{name}]']
    augmented_config = substitute_hyperparameters(baseline_lrsystem_config, substitutions, context)
    lrsystem = parse_lrsystem(
        augmented_config,
        output_dir / f'{dirname_prefix}{name}',
    )
    lrsystem.parameters = hyperparameters
    return lrsystem
