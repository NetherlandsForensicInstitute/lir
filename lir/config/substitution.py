"""
This module offers support functions for replacing/modifying components of an LR Benchmark pipeline
at runtime. For example to compare a logistic regression approach with a support vector approach or to
optimize a given (hyper)parameter of the system.

For example, the `parameters` path of the `model_selection_run` benchmark, which defines the `comparing.clf` as
a path to modify with the options as defined in the `values` section. This will replace (update) the defined
`comparing` module in the LR system configuration, used in this pipeline.
```
benchmarks:
  model_selection_run:
    lr_system: ...
    ...
    parameters:
      - path: comparing.clf
        values:
          - name: logit
            method: logistic_regression
            C: 1
          - name: svm
            method: svm
            probability: True
```
"""

from collections.abc import Iterable, Mapping, Sequence
import itertools
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, NamedTuple

from confidence import Configuration

from lir import registry
from lir.config.base import (
    YamlParseError,
    check_is_empty,
    pop_field,
    config_parser,
)

LOG = logging.getLogger(__name__)


class HyperparameterOption(NamedTuple):
    """
    An option for a value of a hyperparameter.

    A `HyperparameterOption` is a named tuple with two fields:
    - name: a descriptive name of this option
    - substitutions: a mapping of configuration paths to values
    """

    name: str
    substitutions: Mapping[str, Any]

    def __repr__(self) -> str:
        return self.name


class Hyperparameter(ABC):
    """Base class for all hyperparameters."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def options(self) -> list[HyperparameterOption]:
        """
        Get a list of values that a hyperparameter can take in the context of a particular experiment.

        :return: a list of `HyperparameterOption`
        """
        raise NotImplementedError


class CategoricalHyperparameter(Hyperparameter):
    """
    A categorical hyperparameter.

    A categorical hyperparameter has the following fields in a YAML configuration:
    - path: the path of this hyperparameter in the LR system configuration
    - options: a list of options
    """

    def __init__(self, name: str, options: list[HyperparameterOption]):
        super().__init__(name)
        self._options = options

    def options(self) -> list[HyperparameterOption]:
        return self._options


def _parse_categorical_option(
    spec: Any, config_context_path: list[str], path: str, option_index: int | None
) -> HyperparameterOption:
    """
    Parse a section describing an option value of a categorical hyperparameter.
    """

    name = None
    if isinstance(spec, Mapping):
        # use the explicityly declared name, if any
        name = pop_field(config_context_path, spec, "option_name", required=False)

    # use the explicitly declared value, or default to the full subtree
    value = None
    if isinstance(spec, Mapping) and "value" in spec:
        value = pop_field(config_context_path, spec, "value")
    else:
        value = spec

    if name:
        pass
    elif isinstance(value, str):
        name = value
    elif isinstance(value, Mapping) or isinstance(value, list):
        name = f"option{option_index}"
    else:
        name = str(value)

    return HyperparameterOption(name, {path: value})


@config_parser
def parse_categorical(
    spec: dict[str, Any], config_context_path: list[str], output_path: Path
) -> "CategoricalHyperparameter":
    """Parse the `parameters` section of the configuration into a `CategoricalVariable` object."""
    path = pop_field(config_context_path, spec, "path")
    name = pop_field(config_context_path, spec, "name", default=path or "lrsystem")

    # get the option definitions
    options = pop_field(config_context_path, spec, "options")
    options = [
        _parse_categorical_option(spec, config_context_path + ["options", str(i)], path, i)
        for i, spec in enumerate(options)
    ]

    check_is_empty(config_context_path, spec)
    return CategoricalHyperparameter(name, options)


def _parse_substitution(config_context_path: list[str], spec: dict[str, Any]) -> tuple[str, Any]:
    path = pop_field(config_context_path, spec, "path")
    value = pop_field(config_context_path, spec, "value")
    check_is_empty(config_context_path, spec)
    return path, _expand(value)


def _parse_clustered_option(config_context_path: list[str], spec: dict[str, Any]) -> HyperparameterOption:
    option_name = pop_field(config_context_path, spec, "option_name")
    substitutions = pop_field(config_context_path, spec, "substitutions")
    substitutions = [
        _parse_substitution(config_context_path + ["substitutions", str(i)], subst)
        for i, subst in enumerate(substitutions)
    ]
    substitutions = dict(substitutions)
    check_is_empty(config_context_path, spec)
    return HyperparameterOption(option_name, substitutions)


@config_parser
def parse_clustered(
    spec: dict[str, Any], config_context_path: list[str], output_path: Path
) -> CategoricalHyperparameter:
    """
    Parse the configuration section of a clustered hyperparameter.

    A cluster is a set of hyperparameters that are changed at the same time.

    A clustered hyperparameter has the following fields in a YAML configuration:
    - name (optional): a descriptive name for this hyperparameter
    - options: a list of options

    Each option has the following options:
    - name: a descriptive name for this option
    - substitutions: a list of substitutions, with a `path` and `value` field each
    """
    parameter_name = pop_field(config_context_path, spec, "name")
    options = pop_field(config_context_path, spec, "options")
    options = [
        _parse_clustered_option(config_context_path + ["options", str(i)], option) for i, option in enumerate(options)
    ]
    check_is_empty(config_context_path, spec)
    return CategoricalHyperparameter(parameter_name, options)


@config_parser
def parse_constant(
    spec: dict[str, Any], config_context_path: list[str], output_path: Path
) -> CategoricalHyperparameter:
    """
    Parse the configuration section of a constant.

    A constant is functionally identical to a categorical hyperparameter with a single option and has the following
    fields in a YAML configuration:

    - path: the path of this hyperparameter in the LR system configuration
    - value: the substitution value
    """
    path = pop_field(config_context_path, spec, "path")
    value = pop_field(config_context_path, spec, "value")
    value = _parse_categorical_option(value, config_context_path + ["value"], path, 0)
    check_is_empty(config_context_path, spec)
    return CategoricalHyperparameter(path, [value])


class FloatHyperparameter(Hyperparameter):
    """
    A floating point hyperparameter.

    A floating point hyperparameter has the following fields in a YAML configuration:
    - path: the path of this hyperparameter in the LR system configuration
    - low: the lowest possible value
    - high: the highest possible value
    - step (optional): the step size
    - log (optional): if True, search in log space instead of linear space; cannot be combined with `step` (defaults to
      False)
    """

    def __init__(self, path: str, low: float, high: float, step: float | None, log: bool):
        super().__init__(path)
        self.path = path
        self.low = low
        self.high = high
        self.step = step
        self.log = log

    def options(self) -> list[HyperparameterOption]:
        if self.step is None:
            raise ValueError(
                f"unable to generate options for floating point hyperparameter {self.path}: no step size defined"
            )

        n_steps = int((self.high - self.low) // self.step + 1)
        values = [self.low + value * self.step for value in range(n_steps)]
        return [HyperparameterOption(str(value), {self.path: value}) for value in values]


@config_parser
def parse_float(spec: dict[str, Any], config_context_path: list[str], output_path: Path) -> "FloatHyperparameter":
    """Parse the `parameters` section of the configuration into a `CategoricalVariable` object."""
    path = pop_field(config_context_path, spec, "path")
    low = pop_field(config_context_path, spec, "low")
    high = pop_field(config_context_path, spec, "high")
    log = pop_field(config_context_path, spec, "log", default=False)
    step = pop_field(config_context_path, spec, "step", required=False)

    if log and step is not None:
        raise YamlParseError(
            config_context_path,
            "configuration field `log` and `step` cannot be cannot be combined",
        )

    check_is_empty(config_context_path, spec)
    return FloatHyperparameter(path, low, high, step, log)


def _expand(cfg: Any) -> Any:
    """Iteratively unpack the data structure into the appropriate underlying representation."""
    if isinstance(cfg, Mapping):
        return {key: _expand(value) for key, value in cfg.items()}
    elif isinstance(cfg, str):
        return cfg
    elif isinstance(cfg, Sequence):
        return [_expand(value) for value in cfg]
    return cfg


def parse_hyperparameter(
    spec: dict[str, Any],
    config_context_path: list[str],
    output_dir: Path,
) -> Hyperparameter:
    """
    Parse the parameters section of the configuration into a dedicated value wrapper object.
    """

    if "type" in spec:
        parameter_type = pop_field(config_context_path, spec, "type")  # read from specified configuration

        parser = registry.get(parameter_type, search_path=["hyperparameter_types"])
    elif "value" in spec:
        parser = parse_constant()
    elif "options" in spec and "path" in spec:
        parser = parse_categorical()
    elif "options" in spec and "name" in spec:
        parser = parse_clustered()
    elif "high" in spec:
        parser = parse_float()
    else:
        raise YamlParseError(
            config_context_path,
            f"unrecognized hyperparameter type with fields: {', '.join(f'{key}' for key in spec.keys())}",
        )

    return parser.parse(spec, config_context_path, output_dir)


def _assign(struct: dict | list, path: list[str], value: Any) -> None:
    """
    Assigns a new value to a path within an hierarchical `dict` structure.

    Parameters:
        - struct is the `dict` that is modified in-place
        - path is the path within the dict, as a list of `str`
        - value is the value to be assigned
    """
    if isinstance(struct, list):
        index = int(path[0])
        if len(path) == 1:
            struct[index] = value
        else:
            _assign(struct[index], path[1:], value)
    else:
        if len(path) == 1:
            struct[path[0]] = value
        else:
            _assign(struct[path[0]], path[1:], value)


def _path_exists(struct: dict | list, path: list[str]) -> bool:
    index = int(path[0]) if isinstance(struct, list) else path[0]

    if index not in struct:
        if isinstance(struct, dict):
            options = ", ".join(struct.keys())
        else:
            options = f"0..{len(struct) - 1}"
        raise ValueError(f"no such key: {index}; found: {options}")

    if len(path) == 1:
        return index in struct
    else:
        return index in struct and _path_exists(struct[index], path[1:])  # type: ignore


def validate_substitution_paths(
    config_context_path: list[str],
    base_config: Configuration,
    parameter_paths: Iterable[str],
) -> None:
    parameter_paths: list[str | None] = [None] + sorted(parameter_paths)
    for previous_path, path in itertools.pairwise(parameter_paths):
        if path != "" and not _path_exists(_expand(base_config), path.split(".")):  # type: ignore
            raise YamlParseError(config_context_path, f"invalid substitution path: {path}")

        if previous_path is not None:
            if previous_path == "" or path.startswith(previous_path + "."):  # type: ignore[union-attr]
                raise YamlParseError(
                    config_context_path,
                    f"conflicting substitution path: {path} is a sub path of {previous_path}",
                )


def substitute_hyperparameters(base_config: Configuration, hyperparameters: Mapping[str, Any]) -> Configuration:
    """
    Substitute hyperparameters in an LR system configuration and return the updated configuration.

    :param base_config: the original LR system configuration
    :param hyperparameters: the hyperparameters and their values
    :return: the augmented LR system configuration
    """

    if "" in hyperparameters:
        # if the root is assigned, don't bother substituting and return the assigned value immediately
        augmented_config = Configuration(hyperparameters[""])
    else:
        augmented_config = _expand(base_config)
        LOG.debug(f"base system: {json.dumps(augmented_config)}")
        for key, value in hyperparameters.items():
            _assign(augmented_config, key.split("."), value)

    LOG.debug(f"augmented system: {json.dumps(_expand(augmented_config))}")
    return augmented_config
