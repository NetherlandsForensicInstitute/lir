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

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from lir import registry
from lir.config.base import (
    ContextAwareDict,
    ContextAwareList,
    YamlParseError,
    _expand,
    check_is_empty,
    config_parser,
    pop_field,
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


def _parse_categorical_option(spec: Any, path: str, option_index: int | None) -> HyperparameterOption:
    """
    Parse a section describing an option value of a categorical hyperparameter.
    """

    name = None
    if isinstance(spec, Mapping):
        # use the explicityly declared name, if any
        name = pop_field(spec, 'option_name', required=False)

    # use the explicitly declared value, or default to the full subtree
    value = pop_field(spec, 'value') if isinstance(spec, Mapping) and 'value' in spec else spec

    if name:
        pass
    elif isinstance(value, str):
        name = value
    elif isinstance(value, (Mapping, list)):
        name = f'option{option_index}'
    else:
        name = str(value)

    return HyperparameterOption(name, {path: value})


@config_parser
def parse_categorical(spec: ContextAwareDict, output_path: Path) -> 'CategoricalHyperparameter':
    """Parse the `parameters` section of the configuration into a `CategoricalVariable` object."""
    path = pop_field(spec, 'path')
    name = pop_field(spec, 'name', default=path or 'lrsystem')

    # get the option definitions
    options = pop_field(spec, 'options')
    options = [_parse_categorical_option(spec, path, i) for i, spec in enumerate(options)]

    check_is_empty(spec)
    return CategoricalHyperparameter(name, options)


def _parse_substitution(spec: ContextAwareDict) -> tuple[str, Any]:
    path = pop_field(spec, 'path')
    value = pop_field(spec, 'value')
    check_is_empty(spec)
    return path, value


def _parse_clustered_option(spec: ContextAwareDict) -> HyperparameterOption:
    option_name = pop_field(spec, 'option_name')
    substitutions = pop_field(spec, 'substitutions')
    substitutions = [_parse_substitution(subst) for i, subst in enumerate(substitutions)]
    substitutions = dict(substitutions)
    check_is_empty(spec)
    return HyperparameterOption(option_name, substitutions)


@config_parser
def parse_clustered(spec: ContextAwareDict, output_path: Path) -> CategoricalHyperparameter:
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
    parameter_name = pop_field(spec, 'name')
    options = pop_field(spec, 'options')
    options = [_parse_clustered_option(option) for i, option in enumerate(options)]
    check_is_empty(spec)
    return CategoricalHyperparameter(parameter_name, options)


@config_parser
def parse_constant(spec: ContextAwareDict, output_path: Path) -> CategoricalHyperparameter:
    """
    Parse the configuration section of a constant.

    A constant is functionally identical to a categorical hyperparameter with a single option and has the following
    fields in a YAML configuration:

    - path: the path of this hyperparameter in the LR system configuration
    - value: the substitution value
    """
    path = pop_field(spec, 'path')
    value = pop_field(spec, 'value')
    value = _parse_categorical_option(value, path, 0)
    check_is_empty(spec)
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
                f'unable to generate options for floating point hyperparameter {self.path}: no step size defined'
            )

        n_steps = int((self.high - self.low) // self.step + 1)
        values = [self.low + value * self.step for value in range(n_steps)]
        return [HyperparameterOption(str(value), {self.path: value}) for value in values]


@config_parser
def parse_float(spec: ContextAwareDict, output_path: Path) -> 'FloatHyperparameter':
    """Parse the `parameters` section of the configuration into a `CategoricalVariable` object."""
    path = pop_field(spec, 'path')
    low = pop_field(spec, 'low')
    high = pop_field(spec, 'high')
    log = pop_field(spec, 'log', default=False)
    step = pop_field(spec, 'step', required=False)

    if log and step is not None:
        raise YamlParseError(
            spec.context,
            'configuration field `log` and `step` cannot be cannot be combined',
        )

    check_is_empty(spec)
    return FloatHyperparameter(path, low, high, step, log)


class FolderHyperparameter(Hyperparameter):
    """
    A folder hyperparameter that takes all files in a given folder as options.

    A folder hyperparameter has fields in a YAML configuration:
    - folder: the path of the folder containing the options
    - ignore_files: a list of file patterns to ignore

    The generated options will have the full path of each file as both name and value.

    A ValueError can be raised in the following situations:
    - the given folder does not exist
        applies during initialization
    - no valid files are found in the folder (after applying the ignore list)
        applies when calling the `options()` method
    """

    def __init__(self, path: str, folder: str, ignore_files: list[str] | None = None):
        super().__init__(path)

        # Search for the folder in the python PATH. Results in an absolute path.
        folder_path = Path(folder).absolute()

        if not folder_path.is_dir():
            raise ValueError(f'folder hyperparameter {path} points to non-existing folder: {folder}')

        self.folder_path = folder_path

        # Setting ignore files as an empty list if None is given helps avoid checks later on.
        if ignore_files is None:
            self.ignore_files = []
        else:
            self.ignore_files = ignore_files

    def options(self) -> list[HyperparameterOption]:
        """Generates the options by walking over the folder."""
        options = []
        for dirpath, _, filenames in self.folder_path.walk():
            for filename in filenames:
                file = dirpath / filename

                # Check the ignore list patterns
                if not any(file.match(pattern) for pattern in self.ignore_files):
                    options.append(HyperparameterOption(str(filename), {self.name: str(file)}))

        if not options:
            raise ValueError(f'No (valid) files found in folder hyperparameter at path: {self.folder_path}')

        return options


def parse_folder(spec: ContextAwareDict, output_path: Path) -> 'FolderHyperparameter':
    """Parse the `parameters` section of the configuration into a `FolderHyperparameter` object."""
    folder = pop_field(spec, 'folder')
    path = pop_field(spec, 'path')
    ignore_files = pop_field(spec, 'ignore_files', required=False)
    check_is_empty(spec)
    return FolderHyperparameter(path, folder, ignore_files)


def parse_hyperparameter(
    spec: ContextAwareDict,
    output_dir: Path,
) -> Hyperparameter:
    """
    Parse the parameters section of the configuration into a dedicated value wrapper object.
    """

    if 'type' in spec:
        parameter_type = pop_field(spec, 'type')  # read from specified configuration

        parser = registry.get(parameter_type, search_path=['hyperparameter_types'])
    elif 'value' in spec:
        parser = parse_constant()
    elif 'options' in spec and 'path' in spec:
        parser = parse_categorical()
    elif 'options' in spec and 'name' in spec:
        parser = parse_clustered()
    elif 'high' in spec:
        parser = parse_float()
    else:
        raise YamlParseError(
            spec.context,
            f'unrecognized hyperparameter type with fields: {", ".join(f"{key}" for key in spec)}',
        )

    return parser.parse(spec, output_dir)


def _assign(struct: ContextAwareDict | ContextAwareList, path: list[str], value: Any) -> None:
    """
    Assigns a new value to a path within an hierarchical `dict` structure.

    Parameters:
        - struct is the `dict` that is modified in-place
        - path is the path within the dict, as a list of `str`
        - value is the value to be assigned
    """
    if isinstance(struct, list):
        index = int(path[0])
        if index not in struct:
            raise YamlParseError(struct.context, f'trying to substitute invalid index: {index}')
        if len(path) == 1:
            struct[index] = _expand(struct.context + [str(index)], value)
        else:
            _assign(struct[index], path[1:], value)
    else:
        if path[0] not in struct:
            raise YamlParseError(struct.context, f'trying to substitute non-existent field: {path[0]}')
        if len(path) == 1:
            struct[path[0]] = _expand(struct.context + [path[0]], value)
        else:
            _assign(struct[path[0]], path[1:], value)


def _path_exists(struct: dict | list, path: list[str]) -> bool:
    index = int(path[0]) if isinstance(struct, list) else path[0]

    if index not in struct:
        if isinstance(struct, dict):  # noqa: SIM108
            options = ', '.join(struct.keys())
        else:
            options = f'0..{len(struct) - 1}'
        raise ValueError(f'no such key: {index}; found: {options}')

    if len(path) == 1:
        return index in struct
    else:
        return index in struct and _path_exists(struct[index], path[1:])  # type: ignore


def substitute_hyperparameters(
    base_config: ContextAwareDict, hyperparameters: Mapping[str, Any], context: list[str]
) -> ContextAwareDict:
    """
    Substitute hyperparameters in an LR system configuration and return the updated configuration.

    :param base_config: the original LR system configuration
    :param hyperparameters: the hyperparameters and their values
    :param context: the context path of the augmented configuration
    :return: the augmented LR system configuration
    """

    if '' in hyperparameters:
        # if the root is assigned, don't bother substituting and return the assigned value immediately
        augmented_config = _expand(context, hyperparameters[''])
    else:
        LOG.debug(f'base system: {json.dumps(base_config)}')
        augmented_config = base_config.clone(context)
        for key, value in hyperparameters.items():
            try:
                _assign(augmented_config, key.split('.'), value)
            except Exception as e:
                raise ValueError(f'error while trying to substitute {key} in {".".join(base_config.context)}: {e}')

    LOG.debug(f'augmented system: {json.dumps(augmented_config)}')
    return augmented_config
