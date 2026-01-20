import itertools
import logging
from pathlib import Path
from typing import Self

from lir import registry
from lir.config.base import (
    YamlParseError,
    check_is_empty,
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
from lir.data.models import InstanceData, LLRData
from lir.lrsystems.binary_lrsystem import BinaryLRSystem
from lir.lrsystems.lrsystems import LRSystem
from lir.lrsystems.score_based import ScoreBasedSystem
from lir.lrsystems.two_level import TwoLevelSystem
from lir.registry import ComponentNotFoundError


LOG = logging.getLogger(__name__)


class ParsedLRSystem(LRSystem):
    """Represent a given initialized LR system based on the provided configuration."""

    def __init__(self, lrsystem: LRSystem, config: ContextAwareDict, output_dir: Path):
        self.lrsystem = lrsystem
        self.config = config
        self.output_dir = output_dir

    def fit(self, instances: InstanceData) -> Self:
        """Fit the LR system on the instance data."""
        self.lrsystem.fit(instances)
        return self

    def apply(self, instances: InstanceData) -> LLRData:
        """Use the fitted LR system to calculate LLR data for the input instance data."""
        return self.lrsystem.apply(instances)


@config_parser
def specific_source(config: ContextAwareDict, output_dir: Path) -> LRSystem:
    """Construct a specific-source LR system based on the provided configuration.

    The `specific_source` function name corresponds with the naming scheme in the
    registry. See for example: `lir.config.lrsystems.specific_source`.

    The config can contain:
     - modules: module configuration for the pipeline
     - intermediate_output: boolean flag to determine whether to use logging pipeline as default

    If any other fields are present, an exception is raised.
    """
    pipeline = parse_module(
        pop_field(config, 'modules'), output_dir, config.context, default_method=parse_default_pipeline(config)
    )
    check_is_empty(config)
    return BinaryLRSystem(pipeline)


@config_parser
def score_based(config: ContextAwareDict, output_dir: Path) -> LRSystem:
    """Construct a score-based LR system based on the provided configuration.

    The `score_based` function name corresponds with the naming scheme in the
    registry. See for example: `lir.config.lrsystems.score_based`.

    The config can contain:
     - preprocessing: module configuration for preprocessing
     - pairing: module configuration for pairing
     - comparing: module configuration for comparing between scores
     - intermediate_output: boolean flag to determine whether to use logging pipeline as default

    If any other fields are present, an exception is raised.
    """
    default_pipeline = parse_default_pipeline(config)

    preprocessing = parse_module(
        pop_field(config, 'preprocessing'), output_dir, config.context, default_method=default_pipeline
    )
    pairing_function = parse_pairing_config(pop_field(config, 'pairing'), output_dir, config.context)
    evaluation = parse_module(
        pop_field(config, 'comparing'), output_dir, config.context, default_method=default_pipeline
    )

    check_is_empty(config)
    return ScoreBasedSystem(preprocessing, pairing_function, evaluation)


@config_parser
def two_level(config: ContextAwareDict, output_dir: Path) -> TwoLevelSystem:
    """Construct a two-level LR system based on the provided configuration.

    The `two_level` function name corresponds with the naming scheme in the
    registry. See for example: `lir.config.lrsystems.two_level`.

    The config can contain:
    - preprocessing: module for preprocessing trace and reference data
    - pairing: configuration for pairing function
    - postprocessing: module for postprocessing scores to LLRs
    - n_trace_instances: number of trace instances to use
    - n_ref_instances: number of reference instances to use
    - intermediate_output: boolean flag to determine whether to use logging pipeline as default

    If any other fields are present, an exception is raised.
    """
    default_pipeline = parse_default_pipeline(config)

    preprocessing = parse_module(
        pop_field(config, 'preprocessing'), output_dir, config.context, default_method=default_pipeline
    )
    pairing_function = parse_pairing_config(pop_field(config, 'pairing'), output_dir, config.context)
    postprocessing = parse_module(
        pop_field(config, 'postprocessing'), output_dir, config.context, default_method=default_pipeline
    )
    n_trace_instances = pop_field(config, 'n_trace_instances', validate=int)
    n_ref_instances = pop_field(config, 'n_ref_instances', validate=int)

    check_is_empty(config)
    return TwoLevelSystem(
        preprocessing,
        pairing_function,
        postprocessing,
        n_trace_instances,
        n_ref_instances,
    )


def parse_lrsystem(config: ContextAwareDict, output_dir: Path) -> ParsedLRSystem:
    """Determine and initialise corresponding LR system from configuration values.

    LR systems are provided under the `architectures` key.
    """
    lrsystem_config = config.clone()  # save for later

    architecture = pop_field(config, 'architecture')

    try:
        parser = registry.get(architecture, search_path=['lrsystem_architectures'])
    except ComponentNotFoundError as e:
        raise YamlParseError(config.context, f'{e}')

    lrsystem = parser.parse(config, output_dir)
    return ParsedLRSystem(lrsystem, lrsystem_config, output_dir)


def parse_augmented_lrsystem(
    baseline_lrsystem_config: ContextAwareDict,
    hyperparameters: dict[str, HyperparameterOption],
    output_dir: Path,
    dirname_prefix: str = '',
) -> ParsedLRSystem:
    """Parse an augmented LR system.

    The LR system is parsed from a base configuration and a set of parameter substitutions that override parts of the
    base configuration. Results are written to a subdirectory of `output_dir` that is named by its parameter
    substitutions and prefixed by `dirname_prefix`.

    :param baseline_lrsystem_config: the base LR system configuration
    :param hyperparameters: hyperparameter substitutions that override parts of the base configuration
    :param output_dir: the directory where create a results directory
    :param dirname_prefix: the prefix of the created directory name
    :return: the LR system
    """
    substitutions = dict(itertools.chain(*[opt.substitutions.items() for opt in hyperparameters.values()]))

    # generate a descriptive name from the substitutions
    name = '__'.join([f'{key}={value}' for key, value in hyperparameters.items()])

    # generate the YAML context, for debugging and error messages
    context = baseline_lrsystem_config.context + [f'substitution[{name}]']

    # build the augmented configuration for the LR system
    augmented_config = substitute_hyperparameters(baseline_lrsystem_config, substitutions, context)

    # construct and return the LR system
    lrsystem_output_dir = output_dir / f'{dirname_prefix}{name}'
    return parse_lrsystem(augmented_config, lrsystem_output_dir)


def parse_default_pipeline(config: ContextAwareDict) -> str:
    """Parse the intermediate result field from configuration, with the goal of determining the default pipeline method.

    :param config: the configuration dictionary
    :return: the default method to use ('logging_pipeline' or 'pipeline')
    """
    intermediate_output = pop_field(config, 'intermediate_output', default=False)
    default_method = 'logging_pipeline' if intermediate_output else 'pipeline'
    if intermediate_output:
        LOG.debug('Using logging pipeline by default as `intermediate_output` is set to true.')

    return default_method
