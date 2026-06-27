from lir import registry
from lir.config.base import ConfigParser


def _list_components(prefix: str) -> list[str]:
    components = [component[len(prefix):] for component in registry.registry() if component.startswith(prefix)]
    return components


DEFINITIONS = {
    int: {
        "type": "integer",
        "description": "Random seed for reproducibility."
    },
    int: {
        "type": "integer",
        "description": "Random seed for reproducibility."
    },
}


def _generate_attribute_definitions(attributes: None):
    if attributes is None:
        return {}

    attrs = {}
    for attribute_name, attribute_spec in attrs.items():
        if isinstance(attribute_spec, ConfigParser):
            attrs[attribute_name] = {
                "description": "Description (TODO).",
                "$ref": f"#/definitions/{attribute_spec.reference()}",
            }

        attrs[attribute_name] = {}


def generate_experiment_definition():
    strategy_names = _list_components("experiment_strategy.")
    strategy_parsers = [registry.get(name) for name in strategy_names]
    schema = {
        "type": "object",
        "description": "Definition of a single LiR experiment. Experiments are executed independently and produce their own outputs within the configured output path.",
        "properties": {
            "name": {
                "type": "string",
                "minLength": 1,
                "description": "Unique identifier for the experiment. Used in logging, output folder naming, and result tracking.",
            },
            "strategy": {
                "type": "string",
                "description": "Execution strategy controlling how the experiment is run.",
                "enum": strategy_names,
            },
        },
    }

    for i, parser in enumerate(strategy_parsers):
        schema["properties"] |= {
            "if": {
                "properties": {"strategy": {"const": strategy_names[i]}}
            },
            "then": {
                "required": ["data", "lrsystem"]
            },
        }

            "data": {
                "$ref": "#/definitions/dataConfiguration",
                "description": "Data configuration for this experiment (references are resolved before validation)."
            },
            "lrsystem": {
                "oneOf": [
                    {"type": "null",
                     "description": "No LR system specified (will be substituted via lrsystem_parameters)."},
                    {"$ref": "#/definitions/lrSystemConfiguration",
                     "description": "LR system configuration for this experiment."}
                ],
                "description": "LR system configuration for this experiment (references are resolved before validation)."
            },
            "lrsystem_parameters": {
                "type": "array",
                "description": "List of lrsystem_parameters to vary (for grid and optuna strategies).",
                "items": {
                    "$ref": "#/definitions/hyperparameter"
                }
            },
            "data_parameters": {
                "type": "array",
                "description": "List of data parameters to vary (for grid strategy).",
                "items": {
                    "$ref": "#/definitions/hyperparameter"
                }
            },
            "primary_metric": {
                "type": "string",
                "description": "Primary metric to optimize (for optuna strategy). Examples: cllr, cllr_min, cllr_cal."
            },
            "n_trials": {
                "type": "integer",
                "minimum": 1,
                "description": "Number of optimization trials to run (for optuna strategy)."
            },
            "output": {
                "type": "array",
                "minItems": 1,
                "description": "List of output artefacts to generate for this experiment.",
                "items": {
                    "$ref": "#/definitions/outputSpecification"
                }
            },
            "enable_parallelization": {
                "type": "boolean",
                "description": "Whether to run experiments in parallel."
            }
        },
        "required": ["name", "strategy", "output"],
        "allOf": [
            {
                "if": {
                    "properties": {"strategy": {"const": "single_run"}}
                },
                "then": {
                    "required": ["data", "lrsystem"]
                }
            },
            {
                "if": {
                    "properties": {"strategy": {"const": "grid"}}
                },
                "then": {
                    "anyOf": [
                        {"required": ["lrsystem_parameters"]},
                        {"required": ["data_parameters"]}
                    ]
                }
            },
            {
                "if": {
                    "properties": {"strategy": {"const": "optuna"}}
                },
                "then": {
                    "required": ["data", "lrsystem", "lrsystem_parameters", "primary_metric", "n_trials"]
                }
            }
        ],
        "additionalProperties": false
    }

def generate_schema() -> dict:
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "LiR Configuration Schema",
        "description": "Configuration file for running Likelihood Ratio (LiR) experiments. A configuration defines where outputs are written and specifies one or more experiments, each describing data, system configuration, and requested output artefacts.\n\nNote: This configuration uses the 'confidence' library which extends standard YAML with variable substitution using ${variable_name} syntax. This is NOT part of standard YAML but a LiR-specific feature.",
        "type": "object",
        "properties": {
            "output_path": {
                "type": "string",
                "minLength": 1,
                "description": "Base directory where all experiment outputs will be written. The path may contain placeholders such as ${timestamp}, which are resolved at runtime using the confidence library to create unique output folders per run. The ${timestamp} variable is automatically provided by LiR."
            },
            "experiments": {
                "type": "array",
                "minItems": 1,
                "description": "List of experiments to execute. Each experiment defines a LiR pipeline including data selection, LR system configuration, execution strategy, and requested outputs.",
                "items": {
                    "$ref": "#/definitions/experiment"
                }
            }
        },
        "definition": {
            "experiment":
        }
    }
    return schema
