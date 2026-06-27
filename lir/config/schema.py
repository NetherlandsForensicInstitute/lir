import datetime
import types
from pathlib import Path
from typing import Any

from confidence import Configuration, loadf
from jsonschema import validate

from lir import Transformer, registry
from lir.config.base import ConfigAttribute, GenericConfigParser
from lir.config.substitution import Hyperparameter
from lir.experiments.execution import DataConfig, LRSystemConfig
from lir.transform.pairing import PairingMethod
from lir.util import to_native_dict


DEFINITIONS: dict[type, dict[str, Any]] = {
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
    list[Hyperparameter]: {
        'type': 'array',
        'description': 'List of hyperparameters to vary (for grid and optuna strategies).',
        'items': {'$ref': '#/definitions/hyperparameter'},
    },
    types.NoneType: {
        'type': 'null',
    },
}


def _generate_type_definition(config_type: type) -> dict[str, Any]:
    if config_type in DEFINITIONS:
        return DEFINITIONS[config_type]

    if isinstance(config_type, types.UnionType):
        return {'oneOf': [_generate_type_definition(attr) for attr in config_type.__args__]}

    raise ValueError(f'no schema for type: {config_type}')


def _generate_attribute_definition(attribute: ConfigAttribute) -> dict[str, Any]:
    schema = dict(_generate_type_definition(attribute.type))
    if attribute.description is not None:
        schema['description'] = attribute.description
    return schema


def _list_components(prefix: str) -> list[str]:
    components = [component[len(prefix):] for component in registry.registry() if component.startswith(prefix)]
    return components


class SchemaGenerator:
    def __init__(self, extended: bool = False):
        self.extended = extended

    def _generate_alternatives_schema(
            self,
        category: str,
        description: str,
        alternatives_key: str,
        alternatives_desc: str,
        default_alternative: str | None = None,
        allow_shorthand: bool = False,
    ) -> dict[str, Any]:
        config_parsers = {
            name: registry.get(name, default_config_parser=GenericConfigParser, search_path=[category])
            for name in _list_components(f'{category}.')
        }

        schema: dict[str, Any] = {
            'type': 'object',
            'required': [alternatives_key],
            'properties': {
                alternatives_key: {
                    'description': alternatives_desc,
                    'enum': list(config_parsers.keys()),
                },
                'additionalProperties': not all(parser.attributes() is not None for parser in config_parsers.values()),
            },
            'allOf': [],
        }

        for name, parser in config_parsers.items():
            schema['allOf'].append(
                {
                    'if': {'properties': {alternatives_key: {'const': name}}},
                    'then': {
                        'required': [attr.name for attr in parser.attributes() or [] if attr.required],
                        'properties': {
                            attr.name: _generate_attribute_definition(attr) for attr in parser.attributes() or []
                        },
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

        if self.extended and default_alternative is not None:
            parser = registry.get(default_alternative, default_config_parser=GenericConfigParser, search_path=[category])
            options.append(
                {
                    'required': [attr.name for attr in parser.attributes() or [] if attr.required],
                    'properties': {attr.name: _generate_attribute_definition(attr) for attr in parser.attributes() or []},
                }
            )

        if len(options) == 1:
            return options[0] | {
                'description': description,
            }
        else:
            return {
                'description': description,
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
        schema = {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'title': 'LiR Configuration Schema',
            'description': 'Configuration file for running Likelihood Ratio (LiR) experiments. A configuration defines '
            + 'where outputs are written and specifies one or more experiments, each describing data, system '
            + "configuration, and requested output artefacts.\n\nNote: This configuration uses the 'confidence' library "
            + 'which extends standard YAML with variable substitution using ${variable_name} syntax. This is NOT part of '
            + 'standard YAML but a LiR-specific feature.',
            'type': 'object',
            'properties': {
                'output_path': {
                    'type': 'string',
                    'minLength': 1,
                    'description': 'Base directory where all experiment outputs will be written. The path may contain '
                    + 'placeholders such as ${timestamp}, which are resolved at runtime using the confidence library to '
                    + 'create unique output folders per run. The ${timestamp} variable is automatically provided by LiR.',
                },
                'experiments': {
                    'type': 'array',
                    'minItems': 1,
                    'description': 'List of experiments to execute. Each experiment defines a LiR pipeline including data '
                    + 'selection, LR system configuration, execution strategy, and requested outputs.',
                    'items': {'$ref': '#/definitions/experiment'},
                },
            },
            'definitions': {
                'experiment': self._generate_alternatives_schema(
                    'experiment_strategies',
                    'Definition of a single LiR experiment. Experiments are executed independently and produce '
                    + 'their own outputs within the configured output path.',
                    'strategy',
                    'Execution strategy controlling how the experiment is run.',
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
                            'description': 'Data splitting strategy defining how to divide data into training and testing '
                            + 'sets.',
                        },
                    },
                    'required': ['provider', 'splits'],
                    'additionalProperties': False,
                },
                'dataProvider': self._generate_alternatives_schema('data_providers', 'TODO', 'method', 'TODO'),
                'dataSplits': self._generate_alternatives_schema('data_strategies', 'TODO', 'strategy', 'TODO'),
                'lrSystemConfiguration': self._generate_alternatives_schema(
                    'lrsystem_architectures', 'TODO', 'architecture', 'TODO'
                ),
                'module': self._generate_alternatives_schema(
                    'modules', 'TODO', 'method', 'TODO', default_alternative='pipeline', allow_shorthand=True
                ),
                'pairingConfiguration': self._generate_alternatives_schema(
                    'pairing', 'TODO', 'method', 'TODO', allow_shorthand=True
                ),
                'hyperparameter': {
                    'oneOf': [
                        self._generate_alternatives_schema('hyperparameter_types', 'TODO', 'type', 'TODO'),
                        {
                            'properties': {
                                'name': {
                                    'type': 'string',
                                    'description': 'Optional descriptive name for the hyperparameter (defaults to path).',
                                },
                                'path': {
                                    'type': 'string',
                                    'description': "Dot-separated path to the parameter in the configuration (e.g., 'comparing.steps.clf').",
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
                                                        'description': 'Value to substitute (alternative to inline specification).'
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
                        },
                    ],
                },
            },
        }
        return schema


def generate_schema(extended: bool = False) -> dict:
    return SchemaGenerator(extended).generate()


def validate_yaml(yaml_path: Path) -> None:
    """
    Validate a YAML file against the schema.

    Parameters
    ----------
    yaml_path : Path
        The path to the YAML file to be validated.

    Raises
    ------
    FileNotFoundError
        If the YAML file or the schema file does not exist.
    yaml.YAMLError
        If the YAML file is not valid YAML.
    ValidationError
        If the YAML file does not conform to the schema.
    """
    schema = generate_schema()

    # Resolve ${...} references before validation
    context = {'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}  # noqa: DTZ005
    cfg = Configuration(loadf(yaml_path), context)
    data = to_native_dict(cfg)

    # Validate data against schema
    validate(instance=data, schema=schema)
