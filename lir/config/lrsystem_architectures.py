import itertools
import logging
from pathlib import Path
from typing import List, Any

from confidence import Configuration

from lir import registry
from lir.config.base import (
    YamlParseError,
    config_parser,
    parse_pairing_config,
    get_parser_arguments_for_field,
    pop_field,
)
from lir.config.substitution import (
    validate_substitution_paths,
    substitute_hyperparameters,
    HyperparameterOption,
    _expand,
)
from lir.config.transform import parse_module
from lir.lrsystems.lrsystems import LRSystem, Pipeline
from lir.lrsystems.score_based import ScoreBasedSystem
from lir.lrsystems.specific_source import SpecificSourceSystem
from lir.lrsystems.two_level import TwoLevelSystem
from lir.registry import ComponentNotFoundError


LOG = logging.getLogger(__name__)


def parse_pipeline(
    modules_config: Configuration, config_context_path: List[str], output_dir: Path
) -> Pipeline:
    """Construct a scikit-learn Pipeline based on the provided configuration."""
    if modules_config is None:
        return Pipeline([])

    modules = [
        (
            module_name,
            parse_module(
                modules_config.get(module_name),
                config_context_path + [module_name],
                output_dir,
            ),
        )
        for module_name in modules_config
    ]

    return Pipeline(modules)


@config_parser
def specific_source(
    config: dict[str, Any], config_context_path: List[str], output_dir: Path
) -> LRSystem:
    """Construct a specific-source LR system based on the provided configuration.

    The `specific_source` function name corresponds with the naming scheme in the
    registry. See for example: `lir.config.lrsystems.specific_source`.
    """
    pipeline = parse_pipeline(
        *get_parser_arguments_for_field(
            config, config_context_path, output_dir, "modules"
        )
    )
    return SpecificSourceSystem(output_dir.name, pipeline)


@config_parser
def score_based(
    config: dict[str, Any], config_context_path: List[str], output_dir: Path
) -> LRSystem:
    """Construct a specific-source LR system based on the provided configuration.

    The `specific_source` function name corresponds with the naming scheme in the
    registry. See for example: `lir.config.lrsystems.score_based`.
    """
    preprocessing = parse_pipeline(
        *get_parser_arguments_for_field(
            config, config_context_path, output_dir, "preprocessing"
        )
    )
    pairing_function = parse_pairing_config(
        *get_parser_arguments_for_field(
            config, config_context_path, output_dir, "pairing"
        )
    )
    evaluation = parse_pipeline(
        *get_parser_arguments_for_field(
            config, config_context_path, output_dir, "comparing"
        )
    )

    return ScoreBasedSystem(
        output_dir.name, preprocessing, pairing_function, evaluation
    )


@config_parser
def two_level(
    config: dict[str, Any], config_context_path: List[str], output_dir: Path
) -> LRSystem:
    preprocessing = parse_pipeline(
        *get_parser_arguments_for_field(
            config, config_context_path, output_dir, "preprocessing"
        )
    )
    pairing_function = parse_pairing_config(
        *get_parser_arguments_for_field(
            config, config_context_path, output_dir, "pairing"
        )
    )
    postprocessing = parse_pipeline(
        *get_parser_arguments_for_field(
            config, config_context_path, output_dir, "postprocessing"
        )
    )

    n_trace_instances = pop_field(
        config_context_path, config, "n_trace_instances", validate=int
    )
    n_ref_instances = pop_field(
        config_context_path, config, "n_ref_instances", validate=int
    )

    return TwoLevelSystem(
        output_dir.name,
        preprocessing,
        pairing_function,
        postprocessing,
        n_trace_instances,
        n_ref_instances,
    )


def parse_lrsystem(
    config: dict[str, Any], config_context_path: List[str], output_dir: Path
) -> LRSystem:
    """Determine and initialise corresponding LR system from configuration values.

    LR systems are provided under the `architectures` key.
    """
    if not config:
        raise YamlParseError(config_context_path, "empty pipeline definition")

    architecture = pop_field(config_context_path, config, "architecture")

    try:
        parser = registry.get(architecture, search_path=["lrsystem_architectures"])
    except ComponentNotFoundError as e:
        raise YamlParseError(config_context_path, f"{e}")

    return parser.parse(config, config_context_path, output_dir)


def parse_augmented_lrsystem(
    baseline_lrsystem_config: Configuration,
    hyperparameters: dict[str, HyperparameterOption],
    context: List[str],
    output_dir: Path,
    dirname_prefix: str = "",
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
    substitutions = dict(
        itertools.chain(
            *[opt.substitutions.items() for opt in hyperparameters.values()]
        )
    )
    try:
        validate_substitution_paths(
            context, baseline_lrsystem_config, substitutions.keys()
        )
    except Exception as e:
        LOG.warning(f"possible configuration error: {e}")

    augmented_config = substitute_hyperparameters(
        baseline_lrsystem_config, substitutions
    )
    name = "__".join([f"{key}={value}" for key, value in hyperparameters.items()])
    lrsystem = parse_lrsystem(
        _expand(augmented_config),
        context + [f"substitution[{name}]", "lr_system"],
        output_dir / f"{dirname_prefix}{name}",
    )
    lrsystem.parameters = hyperparameters
    return lrsystem
