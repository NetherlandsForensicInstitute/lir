import datetime
import re
import types
from pathlib import Path
from typing import Any

from confidence import Configuration, loadf
from jsonschema import validate

from lir import Transformer, registry
from lir.config.base import ConfigAttribute, ConfigParser, GenericConfigParser
from lir.config.substitution import Hyperparameter
from lir.datasets.synthesized_normal_binary import SynthesizedNormalData
from lir.experiments.execution import DataConfig, LRSystemConfig
from lir.transform.pairing import PairingMethod
from lir.util import to_native_dict


DEFINITIONS: dict[type | types.UnionType, dict[str, Any]] = {
    int: {'type': 'integer', 'description': 'Random seed for reproducibility.'},
    Transformer: {'$ref': '#/definitions/module'},
    PairingMethod: {'$ref': '#/definitions/pairingConfiguration'},
    DataConfig: {
        '$ref': '#/definitions/dataConfiguration',
        'description': 'Data configuration for this experiment (references are resolved before validation).',
    },
    LRSystemConfig: {
        '$ref': '#/definitions/lrSystemConfiguration',
        'description': 'LR system configuration that defines the architecture and modules for likelihood ratio '
        + 'calculation.',
    },
    SynthesizedNormalData: {
        'type': 'object',
        'required': ['mean', 'std', 'size'],
        'properties': {
            'mean': {
                'type': 'number',
                'description': 'Mean value.',
            },
            'std': {
                'type': 'number',
                'description': 'Standard deviation.',
            },
            'size': {
                'type': 'integer',
                'description': 'Number of instances.',
            },
        },
    },
    list[Hyperparameter]: {
        'type': 'array',
        'description': 'List of hyperparameters to vary (for grid and optuna strategies).',
        'items': {'$ref': '#/definitions/hyperparameter'},
    },
    types.NoneType: {
        'type': 'null',
    },
}


def _generate_type_definition(config_type: type | types.UnionType) -> dict[str, Any]:
    if config_type in DEFINITIONS:
        return DEFINITIONS[config_type]

    if isinstance(config_type, types.UnionType):
        return {'oneOf': [_generate_type_definition(attr) for attr in config_type.__args__]}

    if isinstance(config_type, types.GenericAlias) and issubclass(config_type.__origin__, list):
        elem_type = config_type.__args__[0]
        return {'type': 'array', 'items': _generate_type_definition(elem_type)}

    raise ValueError(f'no schema for type: {config_type}')


def _generate_attribute_definition(attribute: ConfigAttribute) -> dict[str, Any]:
    schema = dict(_generate_type_definition(attribute.type))
    if attribute.description is not None:
        schema['description'] = attribute.description
    return schema


def _list_components(prefix: str) -> list[str]:
    components = [component[len(prefix) :] for component in registry.registry() if component.startswith(prefix)]
    return components


def _prune(schema: Any) -> Any:
    """
    Prune a schema by dropping empty elements.

    This creates a copy of the schema from which non-functional elements are removed.

    Parameters
    ----------
    schema : Any
        A schema to prune.

    Returns
    -------
    Any
        The pruned schema.
    """
    if isinstance(schema, dict):
        schema = {key: _prune(value) for key, value in schema.items()}
        if 'required' in schema and len(schema['required']) == 0:
            del schema['required']
        if 'properties' in schema and len(schema['properties']) == 0:
            del schema['properties']
        if 'then' in schema and len(schema['then']) == 0:
            del schema['if']
            del schema['then']
        if 'allOf' in schema and len(schema['allOf']) == 0:
            del schema['allOf']
        if 'oneOf' in schema and len(schema['oneOf']) == 1:
            schema |= _prune(schema['oneOf'][0])
        return schema
    elif isinstance(schema, list):
        schema = [_prune(item) for item in schema]
        schema = [item for item in schema if not (isinstance(item, dict) and len(item) == 0)]
        return schema
    else:
        return schema


def _get_docstr_short(class_name: str | ConfigParser) -> str:
    # TODO: reuse from docs/config.py
    if isinstance(class_name, ConfigParser):
        class_name = class_name.reference()

    docstr = registry._get_attribute_by_name(class_name).__doc__ or ''
    docstr = re.sub('\n.*', '', docstr.strip())
    return docstr


class SchemaGenerator:
    """
    Class for schema generation.

    Parameters
    ----------
    extended : bool
        Allow extended syntax.
    """

    def __init__(self, extended: bool = False):
        self.extended = extended

    def _generate_alternatives_schema(
        self,
        category: str,
        section_description: str,
        method_key: str,
        default_option: str | None = None,
        allow_shorthand: bool = False,
    ) -> dict[str, Any]:
        """
        Generate a schema for a configuration section whose type is a registry section.

        Parameters
        ----------
        category : str
            The name of the registry section. This refers to a section in the registry, i.e. ``registry.yaml``.
        section_description : str
            Description of the section. This should describe the role of the selected option.
        method_key : str
            The key in the configuration section that is used to select an option, e.g. ``method`` for module selection.
        default_option : str | None
            The default value for the variable.
        allow_shorthand : bool
            Allow shorthand-notation.

        Returns
        -------
        dict[str, Any]
            A schema for the configuration section, excluding the section name itself.
        """
        config_parsers = {
            name: registry.get(name, default_config_parser=GenericConfigParser, search_path=[category])
            for name in _list_components(f'{category}.')
        }

        schema: dict[str, Any] = {
            'type': 'object',
            'required': [method_key],
            'properties': {
                method_key: {
                    'enum': list(config_parsers.keys()),
                },
                'additionalProperties': not all(parser.attributes() is not None for parser in config_parsers.values()),
            },
            'allOf': [],
        }

        if default_option is not None:
            schema['properties'][method_key]['default'] = default_option

        for name, parser in config_parsers.items():
            schema['allOf'].append(
                {
                    'if': {'properties': {method_key: {'const': name}}},
                    'then': {
                        'required': [attr.name for attr in parser.attributes() or [] if attr.required],
                        'properties': {'__comment__': {'type': 'null', 'description': _get_docstr_short(parser)}}
                        | {attr.name: _generate_attribute_definition(attr) for attr in parser.attributes() or []},
                    },
                }
            )

        options = [schema]

        if self.extended and allow_shorthand:
            options.append(
                {
                    'type': 'string',
                }
            )

        if self.extended and default_option is not None:
            parser = registry.get(default_option, default_config_parser=GenericConfigParser, search_path=[category])
            options.append(
                {
                    'required': [attr.name for attr in parser.attributes() or [] if attr.required],
                    'properties': {
                        attr.name: _generate_attribute_definition(attr) for attr in parser.attributes() or []
                    },
                }
            )

        if len(options) == 1:
            return options[0] | {
                'description': section_description,
            }
        else:
            return {
                'description': section_description,
                'oneOf': options,
            }

    def generate(self) -> dict[str, Any]:
        """
        Generate a schema for the YAML configuration.

        Returns
        -------
        dict
            The schema as a dict.
        """
        hyperparameter_schema_options = [
            self._generate_alternatives_schema(
                'hyperparameter_types', 'Select the parameter type', 'type', 'categorical'
            )
        ]
        if self.extended:
            hyperparameter_schema_options.append(
                {
                    'properties': {
                        'name': {
                            'type': 'string',
                            'description': 'Optional descriptive name for the hyperparameter (defaults to path).',
                        },
                        'path': {
                            'type': 'string',
                            'description': 'Dot-separated path to the parameter in the configuration (e.g., '
                            + "'comparing.steps.clf').",
                        },
                        'options': {
                            'type': 'array',
                            'description': 'List of categorical options or clustered substitutions.',
                            'items': {
                                'oneOf': [
                                    {'type': 'string'},
                                    {'type': 'number'},
                                    {'type': 'boolean'},
                                    {
                                        'type': 'object',
                                        'properties': {
                                            'option_name': {
                                                'type': 'string',
                                                'description': 'Name for this option.',
                                            },
                                            'value': {
                                                'description': 'Value to substitute (alternative to inline '
                                                + 'specification).'
                                            },
                                            'method': {
                                                'type': 'string',
                                                'description': 'Method name (when option is a module specification).',
                                            },
                                            'substitutions': {
                                                'type': 'array',
                                                'description': 'List of path/value substitutions (for cluster type).',
                                                'items': {
                                                    'type': 'object',
                                                    'properties': {'path': {'type': 'string'}, 'value': {}},
                                                    'required': ['path', 'value'],
                                                    'additionalProperties': False,
                                                },
                                            },
                                        },
                                        'additionalProperties': True,
                                    },
                                ]
                            },
                        },
                        'low': {'type': 'number', 'description': 'Lower bound for float hyperparameter.'},
                        'high': {'type': 'number', 'description': 'Upper bound for float hyperparameter.'},
                        'step': {
                            'type': 'number',
                            'description': 'Step size for float hyperparameter (for grid search).',
                        },
                        'log': {
                            'type': 'boolean',
                            'description': 'Sample from log space instead of linear (for float hyperparameter).',
                        },
                        'folder': {
                            'type': 'string',
                            'description': 'Path to folder containing options (for folder hyperparameter type).',
                        },
                        'ignore_files': {
                            'type': 'array',
                            'description': 'File patterns to ignore in folder (for folder hyperparameter type).',
                            'items': {'type': 'string'},
                        },
                        'value': {'description': 'Constant value (for constant hyperparameter type).'},
                    },
                    'additionalProperties': False,
                }
            )

        schema = {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'title': 'LiR Configuration Schema',
            'description': 'Configuration file for running Likelihood Ratio (LiR) experiments. A configuration defines '
            + 'where outputs are written and specifies one or more experiments, each describing data, system '
            + "configuration, and requested output artefacts.\n\nNote: This configuration uses the 'confidence' library"
            + ' which extends standard YAML with variable substitution using ${variable_name} syntax. This is NOT part '
            + 'of standard YAML but a LiR-specific feature.',
            'type': 'object',
            'properties': {
                'output_path': {
                    'type': 'string',
                    'minLength': 1,
                    'description': 'Base directory where all experiment outputs will be written. The path may '
                    + 'contain placeholders such as ${timestamp}, which are resolved at runtime using the confidence '
                    + 'library to create unique output folders per run. The ${timestamp} variable is automatically '
                    + 'provided by LiR.',
                },
                'experiments': {
                    'type': 'array',
                    'minItems': 1,
                    'description': 'List of experiments to execute. Each experiment defines a LiR pipeline '
                    + 'including data selection, LR system configuration, execution strategy, and requested outputs. '
                    + 'Experiments are executed independently and produce their own outputs within '
                    + 'the configured output path.',
                    'items': {'$ref': '#/definitions/experiment'},
                },
            },
            'definitions': {
                'experiment': self._generate_alternatives_schema(
                    'experiment_strategies',
                    'Define a LiR experiment. First, choose the experiment execution strategy, controlling how the '
                    + 'experiment is run.',
                    'strategy',
                    'single_run',
                ),
                'dataConfiguration': {
                    'type': 'object',
                    'description': 'Configuration for data loading and splitting into train/test sets.',
                    'properties': {
                        'provider': {
                            '$ref': '#/definitions/dataProvider',
                            'description': 'Data provider configuration specifying the data source and how to load it.',
                        },
                        'splits': {
                            '$ref': '#/definitions/dataSplits',
                            'description': 'Data splitting strategy defining how to divide data into training and '
                            + 'testing sets.',
                        },
                    },
                    'required': ['provider', 'splits'],
                    'additionalProperties': False,
                },
                'dataProvider': self._generate_alternatives_schema(
                    'data_providers', 'The data provider specifies the data source and how to load it.', 'method'
                ),
                'dataSplits': self._generate_alternatives_schema(
                    'data_strategies',
                    'Choose a data splitting strategy to define how to divide data into training and '
                    + 'testing sets.',
                    'strategy',
                ),
                'lrSystemConfiguration': self._generate_alternatives_schema(
                    'lrsystem_architectures', 'Choose an LR system architecture.', 'architecture'
                ),
                'module': self._generate_alternatives_schema(
                    'modules',
                    'Select the module to run. Multiple modules can be combined in a pipeline. The '
                    + 'identity module does nothing and can be used as a placeholder.',
                    'method',
                    default_option='identity',
                    allow_shorthand=True,
                ),
                'pairingConfiguration': self._generate_alternatives_schema(
                    'pairing',
                    'Select the pairing method, which controls how instances are combined into pairs.',
                    'method',
                    allow_shorthand=True,
                ),
                'hyperparameter': {
                    'oneOf': hyperparameter_schema_options,
                },
            },
        }
        return _prune(schema)


def generate_schema(extended: bool = False) -> dict[str, Any]:
    """
    Generate a schema for LiR experiment definition.

    Parameters
    ----------
    extended : bool
        Allow extended syntax (recommended for validation, not for form generation).

    Returns
    -------
    dict
        A schema.
    """
    return SchemaGenerator(extended).generate()


def validate_yaml(yaml_path: Path, strict: bool = False) -> None:
    """
    Validate a YAML file against the schema.

    In strict mode, only the canonical form is allowed. This is generally recommended if the use of a GUI is intended.

    Parameters
    ----------
    yaml_path : Path
        The path to the YAML file to be validated.
    strict : bool, optional
        Strict mode.

    Raises
    ------
    FileNotFoundError
        If the YAML file or the schema file does not exist.
    yaml.YAMLError
        If the YAML file is not valid YAML.
    ValidationError
        If the YAML file does not conform to the schema.
    """
    schema = generate_schema(extended=not strict)

    # Resolve ${...} references before validation
    context = {'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}  # noqa: DTZ005
    cfg = Configuration(loadf(yaml_path), context)
    data = to_native_dict(cfg)

    # Validate data against schema
    validate(instance=data, schema=schema)
