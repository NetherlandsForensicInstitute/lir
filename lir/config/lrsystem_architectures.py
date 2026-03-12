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
    substitute_parameters,
)
from lir.config.transform import parse_module
from lir.data.models import InstanceData, LLRData
from lir.lrsystems.binary_lrsystem import BinaryLRSystem
from lir.lrsystems.lrsystems import LRSystem
from lir.lrsystems.score_based import Pipeline, ScoreBasedSystem
from lir.lrsystems.two_level import TwoLevelSystem
from lir.registry import ComponentNotFoundError
from lir.util import check_type


LOG = logging.getLogger(__name__)


class ParsedLRSystem(LRSystem):
    """
    Represent a given initialized LR system based on the provided configuration.

    Parameters
    ----------
    lrsystem : LRSystem
        Underlying LR system implementation.
    config : ContextAwareDict
        Original LR system configuration.
    """

    def __init__(self, lrsystem: LRSystem, config: ContextAwareDict):
        self.lrsystem = lrsystem
        self.config = config

    def fit(self, instances: InstanceData) -> Self:
        """
        Fit the LR system on instance data.

        Parameters
        ----------
        instances : InstanceData
            Training instances.

        Returns
        -------
        Self
            This wrapper instance.
        """
        self.lrsystem.fit(instances)
        return self

    def apply(self, instances: InstanceData) -> LLRData:
        """
        Apply the fitted LR system.

        Parameters
        ----------
        instances : InstanceData
            Instances to score.

        Returns
        -------
        LLRData
            Computed likelihood-ratio data.
        """
        return self.lrsystem.apply(instances)


@config_parser
def specific_source(config: ContextAwareDict, output_dir: Path) -> BinaryLRSystem:
    """
    Construct a specific-source LR system based on the provided configuration.

    The `specific_source` function name corresponds with the naming scheme in the
    registry. See for example: `lir.config.lrsystems.specific_source`.

    The config can contain:
     - modules: module configuration for the pipeline
     - save_features_after_step: optional dict mapping field names to pipeline step names
     - intermediate_output: boolean flag to determine whether to use logging pipeline as default

    If any other fields are present, an exception is raised.

    Parameters
    ----------
    config : ContextAwareDict
        Specific-source architecture configuration.
    output_dir : Path
        Output directory passed to nested module parsers.

    Returns
    -------
    BinaryLRSystem
        Configured specific-source LR system.
    """
    save_features_after_step = pop_field(config, 'save_features_after_step', required=False, validate=dict)
    pipeline = parse_module(
        pop_field(config, 'modules'), output_dir, config.context, default_method=parse_default_pipeline(config)
    )
    pipeline = check_type(Pipeline, pipeline)
    check_is_empty(config)
    return BinaryLRSystem(pipeline, save_features_after_step)


@config_parser
def score_based(config: ContextAwareDict, output_dir: Path) -> ScoreBasedSystem:
    """
    Construct a score-based LR system based on the provided configuration.

    The `score_based` function name corresponds with the naming scheme in the
    registry. See for example: `lir.config.lrsystems.score_based`.

    The config can contain:
     - preprocessing: module configuration for preprocessing
     - pairing: module configuration for pairing
     - comparing: module configuration for comparing between scores
     - intermediate_output: boolean flag to determine whether to use logging pipeline as default

    If any other fields are present, an exception is raised.

    Parameters
    ----------
    config : ContextAwareDict
        Score-based architecture configuration.
    output_dir : Path
        Output directory passed to nested module parsers.

    Returns
    -------
    ScoreBasedSystem
        Configured score-based LR system.
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
    """
    Construct a two-level LR system based on the provided configuration.

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

    Parameters
    ----------
    config : ContextAwareDict
        Two-level architecture configuration.
    output_dir : Path
        Output directory passed to nested module parsers.

    Returns
    -------
    TwoLevelSystem
        Configured two-level LR system.
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
    """
    Determine and initialise corresponding LR system from configuration values.

    LR systems are provided under the `architectures` key.

    Parameters
    ----------
    config : ContextAwareDict
        LR system configuration.
    output_dir : Path
        Output directory for nested parser calls.

    Returns
    -------
    ParsedLRSystem
        Wrapper containing parsed LR system and source configuration.
    """
    lrsystem_config = config.clone()  # save for later

    architecture = pop_field(config, 'architecture')

    try:
        parser = registry.get(architecture, search_path=['lrsystem_architectures'])
    except ComponentNotFoundError as e:
        raise YamlParseError(config.context, f'{e}')

    lrsystem = parser.parse(config, output_dir)
    return ParsedLRSystem(lrsystem, lrsystem_config)


def augment_config(
    baseline_config: ContextAwareDict, hyperparameters: dict[str, HyperparameterOption]
) -> ContextAwareDict:
    """
    Parse an augmented LR system.

    The LR system is parsed from a base configuration and a set of parameter substitutions that override parts of the
    base configuration. Results are written to a subdirectory of `output_dir` that is named by its parameter
    substitutions and prefixed by `dirname_prefix`.

    Parameters
    ----------
    baseline_config : ContextAwareDict
        Base LR system configuration.
    hyperparameters : dict[str, HyperparameterOption]
        Hyperparameter substitutions overriding parts of the base configuration.

    Returns
    -------
    ContextAwareDict
        Augmented LR system configuration.
    """
    substitutions = dict(itertools.chain(*[opt.substitutions.items() for opt in hyperparameters.values()]))

    # generate a descriptive name from the substitutions
    name = '__'.join([f'{key}={value}' for key, value in hyperparameters.items()])

    # generate the YAML context, for debugging and error messages
    context = baseline_config.context + [f'substitution[{name}]']

    # build the augmented configuration for the LR system
    augmented_config = substitute_parameters(baseline_config, substitutions, context)

    return augmented_config


def parse_default_pipeline(config: ContextAwareDict) -> str:
    """
    Parse the intermediate output flag to determine the default pipeline method.

    Parameters
    ----------
    config : ContextAwareDict
        Configuration dictionary.

    Returns
    -------
    str
        Default method name (``'logging_pipeline'`` or ``'pipeline'``).
    """
    intermediate_output = pop_field(config, 'intermediate_output', default=False)
    default_method = 'logging_pipeline' if intermediate_output else 'pipeline'
    if intermediate_output:
        LOG.debug('Using logging pipeline by default as `intermediate_output` is set to true.')

    return default_method
