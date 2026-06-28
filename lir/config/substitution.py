"""
Substitution module.

This module provides utility functions for replacing or modifying components of
an LR Benchmark pipeline at runtime. Typical use cases include comparing
different modelling approaches (e.g. logistic regression versus support vector
machines) or optimising system lrsystem_parameters.

For example, the ``parameters`` section of the ``model_selection_run`` benchmark
can define a path (``comparing.clf``) to be modified using the options listed in
the ``values`` field. Each option updates the ``comparing`` component in the LR
system configuration used by the pipeline.

.. code-block:: yaml

    experiments:
      - name: model_selection_run
        lrsystem: ...
        ...
        lrsystem_parameters:
          - path: comparing.clf
            options:
              - name: logit
                method: logistic_regression
                C: 1
              - name: svm
                method: svm
                probability: True
"""  # noqa: D214, D405, D406, D407, D411

import json
import logging
import numbers
from abc import ABC, abstractmethod
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

from lir import registry
from lir.config.base import (
    ConfigParser,
    ConfigValue,
    YamlParseError,
    check_is_empty,
    config_parser,
    pop_field,
)
from lir.data.io import search_path
from lir.util import check_type


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
    """
    Base class for all lrsystem_parameters.

    Parameters
    ----------
    name : str
        Hyperparameter name.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def options(self) -> list[HyperparameterOption]:
        """
        Get a list of values that a hyperparameter can take in the context of a particular experiment.

        Returns
        -------
        list[HyperparameterOption]
            List of options for this hyperparameter.
        """
        raise NotImplementedError


class CategoricalHyperparameter(Hyperparameter):
    """
    A categorical hyperparameter.

    A categorical hyperparameter has the following fields in a YAML configuration:
    - path: the path of this hyperparameter in the LR system configuration
    - options: a list of options

    Parameters
    ----------
    name : str
        Hyperparameter name.
    options : list[HyperparameterOption]
        Available options.
    """

    def __init__(self, name: str, options: list[HyperparameterOption]):
        super().__init__(name)
        self._options = options

    def options(self) -> list[HyperparameterOption]:
        """
        Provide API access to the options for the hyperparameter.

        Returns
        -------
        list[HyperparameterOption]
            Configured categorical options.
        """
        return self._options


def _parse_categorical_option(spec: Any, path: str, option_index: int | None) -> HyperparameterOption:
    """
    Parse one categorical option specification.

    An option generally has a name and a value. The name is a human-readable string. This is how it is referred to is
    user output. The value is a number or a string, this is usually a sensible way to refer to the option, and there is
    no need to define a name explicitly. If the value is more complex, like a dictionary, a more friendly name can be
    defined.

    The option can have one of several formats.

    The simplest way is to give just the values. This is a good choice if the values are strings or numbers:

    .. code-block:: yaml

        options:
          - some_value
          - some_other_value

    The same notation is also valid for more complex values, in which case the options are referred to as `option0`,
    `option1`, etc., instead of the full tree:

    .. code-block:: yaml

        options:
          - method: logistic_regression
            C: 1
          - method: svm
            probability: True

    To avoid non-informative names, the option's name can be declared explicitly, like so:

    .. code-block:: yaml

        options:
          - option_name: logit
            method: logistic_regression
            C: 1
          - option_name: svm
            method: svm
            probability: True

    The value can also be declared explicitly. The following snippet is equivalent to the previous one:

    .. code-block:: yaml

        options:
          - option_name: logit
            value:
              method: logistic_regression
              C: 1
          - option_name: svm
            value:
              method: svm
              probability: True

    Parameters
    ----------
    spec : Any
        Option specification.
    path : str
        Target substitution path.
    option_index : int | None
        Position of this option in the list.

    Returns
    -------
    HyperparameterOption
        Parsed option.
    """
    # check if the option specification is a dictionary
    if isinstance(spec.value, dict):
        # use the explicitly declared name and value
        name = pop_field(spec, 'option_name', required=False, validate_type=str)
        if 'value' in spec:
            value = pop_field(spec, 'value', unwrap=True)
            check_is_empty(spec)
        else:
            value = spec.unwrap()
    else:
        # default to the full subtree for `value` and worry about `name` later
        name = None
        value = spec.unwrap()

    if not name:
        if isinstance(value, str):
            # the value is a str, which is usually a good default description
            name = value
        elif isinstance(value, (Mapping, list)):
            # the value is a complex data structure -> default to a generic name with an index number
            name = f'option{option_index}'
        else:
            # something else -> try to convert it to a string
            name = str(value)

    return HyperparameterOption(name, {path: value})


@config_parser(reference='lir.config.substitution.parse_categorical')
def parse_categorical(spec: ConfigValue, output_path: Path) -> 'CategoricalHyperparameter':
    """
    Parse a categorical hyperparameter from configuration.

    Parameters
    ----------
    spec : ConfigValue
        Hyperparameter specification.
    output_path : Path
        Unused output path required by parser API.

    Returns
    -------
    CategoricalHyperparameter
        Parsed categorical hyperparameter.
    """
    path = pop_field(spec, 'path', validate_type=str)
    name = pop_field(spec, 'name', default=path or 'lrsystem')

    # get the option definitions
    options = pop_field(spec, 'options')
    options = [_parse_categorical_option(option_config, path, i) for i, option_config in enumerate(options)]

    check_is_empty(spec)
    return CategoricalHyperparameter(name, options)


def _parse_substitution(spec: ConfigValue) -> tuple[str, Any]:
    """
    Parse one substitution specification.

    Parameters
    ----------
    spec : ConfigValue
        Substitution specification with ``path`` and ``value`` fields.

    Returns
    -------
    tuple[str, Any]
        Path and value pair.
    """
    path = pop_field(spec, 'path', validate_type=str)
    value = pop_field(spec, 'value', unwrap=True)
    check_is_empty(spec)
    return path, value


def _parse_clustered_option(spec: ConfigValue) -> HyperparameterOption:
    option_name = pop_field(spec, 'option_name', validate=str)
    substitutions = pop_field(spec, 'substitutions', unwrap=False)
    substitutions.check_type(list)
    substitutions = [_parse_substitution(subst) for i, subst in enumerate(substitutions)]
    substitutions = dict(substitutions)
    check_is_empty(spec)
    return HyperparameterOption(option_name, substitutions)


@config_parser(reference='lir.config.substitution.parse_clustered')
def parse_clustered(spec: ConfigValue, output_path: Path) -> CategoricalHyperparameter:
    """
    Parse the configuration section of a clustered hyperparameter.

    A cluster is a set of lrsystem_parameters that are changed at the same time.

    A clustered hyperparameter has the following fields in a YAML configuration:
    - name (optional): a descriptive name for this hyperparameter
    - options: a list of options

    Each option has the following options:
    - name: a descriptive name for this option
    - substitutions: a list of substitutions, with a `path` and `value` field each

    Parameters
    ----------
    spec : ConfigValue
        Hyperparameter specification.
    output_path : Path
        Unused output path required by parser API.

    Returns
    -------
    CategoricalHyperparameter
        Parsed clustered hyperparameter.
    """
    parameter_name = pop_field(spec, 'name', validate=str)
    options = pop_field(spec, 'options')
    options = [_parse_clustered_option(option) for i, option in enumerate(options)]
    check_is_empty(spec)
    return CategoricalHyperparameter(parameter_name, options)


@config_parser(reference='lir.config.substitution.parse_constant')
def parse_constant(spec: ConfigValue, output_path: Path) -> CategoricalHyperparameter:
    """
    Parse the configuration section of a constant.

    A constant is functionally identical to a categorical hyperparameter with a single option and has the following
    fields in a YAML configuration:

    - path: the path of this hyperparameter in the LR system configuration
    - value: the substitution value

    Parameters
    ----------
    spec : ConfigValue
        Hyperparameter specification.
    output_path : Path
        Unused output path required by parser API.

    Returns
    -------
    CategoricalHyperparameter
        Parsed constant as a single-option categorical hyperparameter.
    """
    path = pop_field(spec, 'path', validate_type=str)
    value = pop_field(spec, 'value')
    value = _parse_categorical_option(value, path, 0)
    check_is_empty(spec)
    return CategoricalHyperparameter(path, [value])


class FloatHyperparameter(Hyperparameter):
    """
    Floating-point hyperparameter.

    In a YAML configuration, this hyperparameter supports the following fields:

    - ``path``: Path to the hyperparameter in the LR system configuration.
    - ``low``: Lower bound of the search range.
    - ``high``: Upper bound of the search range.
    - ``step`` (optional): Step size for a linear grid search.
    - ``log`` (optional): If ``True``, search in logarithmic space instead of
      linear space. Cannot be combined with ``step``. Defaults to ``False``.

    Parameters
    ----------
    path : str
        Configuration path to substitute.
    low : float
        Lower bound.
    high : float
        Upper bound.
    step : float | None
        Optional step size for grid options.
    log : bool
        Whether to sample in log space.
    """

    def __init__(self, path: str, low: float, high: float, step: float | None, log: bool):
        super().__init__(path)
        self.path = path
        self.low = low
        self.high = high
        self.step = step
        self.log = log

    def options(self) -> list[HyperparameterOption]:
        """
        Provide API access to the options for the hyperparameter.

        Returns
        -------
        list[HyperparameterOption]
            Enumerated hyperparameter options.
        """
        if self.step is None:
            raise ValueError(
                f'unable to generate options for floating point hyperparameter {self.path}: no step size defined'
            )

        n_steps = int((self.high - self.low) // self.step + 1)
        values = [self.low + value * self.step for value in range(n_steps)]
        return [HyperparameterOption(str(value), {self.path: value}) for value in values]


@config_parser(reference='lir.config.substitution.parse_float')
def parse_float(spec: ConfigValue, output_path: Path) -> 'FloatHyperparameter':
    """
    Parse a floating-point hyperparameter from configuration.

    Parameters
    ----------
    spec : ConfigValue
        Hyperparameter specification.
    output_path : Path
        Unused output path required by parser API.

    Returns
    -------
    FloatHyperparameter
        Parsed floating-point hyperparameter.
    """
    path = pop_field(spec, 'path', validate_type=str)
    low = pop_field(spec, 'low', validate_type=numbers.Number)
    high = pop_field(spec, 'high', validate_type=numbers.Number)
    log = pop_field(spec, 'log', default=False, validate_type=bool)
    step = pop_field(spec, 'step', required=False, validate_type=numbers.Number)

    if log and step is not None:
        raise YamlParseError(
            spec.context,
            'configuration field `log` and `step` cannot be combined',
        )

    check_is_empty(spec)
    return FloatHyperparameter(path, low, high, step, log)


class FolderHyperparameter(Hyperparameter):
    """
    Hyperparameter that enumerates all files in a given folder as options.

    This hyperparameter reads the contents of a specified folder and generates one
    option per file. Each option uses the file’s full path as both its name and its
    value.

    In a YAML configuration, a folder hyperparameter supports the following fields:

    - ``folder``: Path to the folder containing the candidate files.
    - ``ignore_files``: Optional list of file patterns to ignore.

    Example configuration:

    .. code-block:: yaml

        lrsystem_parameters:
        - path: data.provider.path
          type: folder
          folder: project_files/my_dataset/
          ignore_files:  # Optional list of file patterns to ignore.
           - '*.tmp'
           - 'ignore_this_file.csv'

    Parameters
    ----------
    path : str
        Configuration path to substitute.
    folder : str
        Folder containing candidate files.
    ignore_files : list[str] | None, optional
        Filename patterns to exclude.

    Raises
    ------
    ValueError
        If the specified folder does not exist (during initialisation).
    ValueError
        If no valid files are found in the folder after applying the ignore
        patterns (when calling :meth:`options`).
    """

    def __init__(self, path: str, folder: str, ignore_files: list[str] | None = None):
        super().__init__(path)

        # Search for the folder in the python PATH. Results in an absolute path.
        folder_path = search_path(Path(folder))

        if not folder_path.is_dir():
            raise ValueError(f'folder hyperparameter {path} points to non-existing folder: {folder}')

        self.folder_path = folder_path

        # Setting ignore files as an empty list if None is given helps avoid checks later on.
        self.ignore_files = ignore_files if ignore_files is not None else []

    def options(self) -> list[HyperparameterOption]:
        """
        Generate options by walking over the folder.

        Returns
        -------
        list[HyperparameterOption]
            File-based options discovered in the folder.
        """
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


@config_parser(reference='lir.config.substitution.parse_folder')
def parse_folder(spec: ConfigValue, output_path: Path) -> 'FolderHyperparameter':
    """
    Parse a folder hyperparameter from configuration.

    Parameters
    ----------
    spec : ConfigValue
        Hyperparameter specification.
    output_path : Path
        Unused output path required by parser API.

    Returns
    -------
    FolderHyperparameter
        Parsed folder hyperparameter.
    """
    folder = pop_field(spec, 'folder', validate_type=str)
    path = pop_field(spec, 'path', validate_type=str)
    ignore_files = pop_field(spec, 'ignore_files', required=False, validate=partial(check_type, list))
    check_is_empty(spec)
    return FolderHyperparameter(path, folder, ignore_files)


def parse_parameter(
    spec: ConfigValue,
    output_dir: Path,
) -> Hyperparameter:
    """
    Parse one parameter specification into a hyperparameter object.

    Parameters
    ----------
    spec : ConfigValue
        Parameter specification.
    output_dir : Path
        Output directory used by nested parser calls.

    Returns
    -------
    Hyperparameter
        Parsed hyperparameter object.
    """
    parser: ConfigParser
    if 'type' in spec:
        # read from specified configuration
        parameter_type = pop_field(spec, 'type', validate_type=str)

        parser = registry.get(parameter_type, search_path=['hyperparameter_types'])
    elif 'value' in spec:
        parser = parse_constant()  # type: ignore
    elif 'options' in spec and 'path' in spec:
        parser = parse_categorical()  # type: ignore
    elif 'options' in spec and 'name' in spec:
        parser = parse_clustered()  # type: ignore
    elif 'high' in spec:
        parser = parse_float()  # type: ignore
    else:
        raise YamlParseError(
            spec.context,
            f'unrecognized hyperparameter type with fields: {", ".join(f"{key}" for key in spec)}',
        )

    return parser.parse(spec, output_dir)


def parse_config_with_parameters(
    config: ConfigValue,
    output_dir: Path,
    config_field: str,
    parameters_field: str,
) -> tuple[ConfigValue, list[Hyperparameter]]:
    """
    Extract a configuration section and its associated parameters.

    Parameters
    ----------
    config : ConfigValue
        The configuration.
    output_dir : Path
        The output directory.
    config_field : str
        Field containing the baseline configuration.
    parameters_field : str
        Field containing parameters to vary.

    Returns
    -------
    tuple[ConfigValue, list[Hyperparameter]]
        Baseline configuration and parsed hyperparameters.
    """
    baseline_config = pop_field(config, config_field, default=ConfigValue(config.context + [config_field], {}))

    parameters = []
    if parameters_field in config:
        parameters = pop_field(config, parameters_field)
        parameters = [parse_parameter(variable, output_dir) for variable in parameters]

    return baseline_config, parameters


def _assign(struct: ConfigValue, path: list[str], value: Any) -> None:
    """
    Assign a new value to a path within an hierarchical `dict` structure.

    Parameters
    ----------
    struct : ConfigValue
        Structure that is modified in-place.
    path : list[str]
        Path within the structure.
    value : Any
        Value to assign.

    Returns
    -------
    None
        This function mutates ``struct`` in-place.
    """
    if isinstance(struct.value, list):
        index = int(path[0])
        if index not in struct:
            raise YamlParseError(struct.context, f'trying to substitute invalid index: {index}')
        if len(path) == 1:
            struct[index] = ConfigValue.wrap(struct.context + [str(index)], value)
        else:
            _assign(struct[index], path[1:], value)
    elif isinstance(struct.value, dict):
        if path[0] not in struct:
            raise YamlParseError(struct.context, f'trying to substitute non-existent field: {path[0]}')
        if len(path) == 1:
            struct[path[0]] = ConfigValue.wrap(struct.context + [path[0]], value)
        else:
            _assign(struct[path[0]], path[1:], value)
    else:
        raise YamlParseError(struct.context, 'illegal state')


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


def substitute_parameters(
    base_config: ConfigValue, lrsystem_parameters: Mapping[str, Any], context: list[str]
) -> ConfigValue:
    """
    Substitute parameters in an LR system configuration and return the updated configuration.

    Parameters
    ----------
    base_config : ConfigValue
        Original LR system configuration.
    lrsystem_parameters : Mapping[str, Any]
        LR system parameters to vary and their replacement values.
    context : list[str]
        Context path of the augmented configuration.

    Returns
    -------
    ConfigValue
        Augmented LR system configuration.
    """
    if '' in lrsystem_parameters:
        # if the root is assigned, don't bother substituting and return the assigned value immediately
        augmented_config = ConfigValue.wrap(context, lrsystem_parameters[''])
    else:
        LOG.debug(f'base system: {json.dumps(base_config.unwrap())}')
        augmented_config = base_config.clone(context)
        for key, value in lrsystem_parameters.items():
            try:
                _assign(augmented_config, key.split('.'), value)
            except Exception as e:
                raise ValueError(f'error while trying to substitute {key} in {".".join(base_config.context)}: {e}')

    LOG.debug(f'augmented system: {json.dumps(augmented_config.unwrap())}')
    return augmented_config
