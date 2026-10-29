# test_yaml_schema.py
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from jsonschema import ValidationError, validate

from lir.config.schema import generate_schema, validate_yaml


ROOT = Path(__file__).resolve().parent.parent  # Go up to project root
SCHEMA_PATH = ROOT / 'lir.schema.json'
YAML_DIR = ROOT / 'examples'


@pytest.mark.parametrize('yaml_path', sorted(YAML_DIR.glob('**/*.y*ml')))
def test_yaml_files_conform_to_schema(yaml_path: Path) -> None:
    """Test that all YAML files in the examples directory conform to the schema."""
    try:
        validate_yaml(yaml_path)
    except (FileNotFoundError, yaml.YAMLError, ValidationError) as e:
        pytest.fail(f'Validation failed for {yaml_path}: {e}')


def test_data_section_without_setup_is_valid() -> None:
    """A missing `setup` key is accepted by the schema, matching the parser default."""
    instance = {
        'output_path': 'output',
        'experiments': [
            {
                'name': 'exp',
                'strategy': 'single_run',
                'lrsystem': {
                    'architecture': 'specific_source',
                    'modules': {'steps': {'clf': 'logistic_regression', 'to_llr': 'probabilities_to_logodds'}},
                },
                'data': {
                    'provider': {
                        'method': 'synthesized_normal_binary',
                        'h1': {'mean': 0, 'std': 1, 'size': 10},
                        'h2': {'mean': 2, 'std': 1, 'size': 10},
                    },
                    'splits': {'strategy': 'train_test_instances'},
                },
                'output': [{'method': 'metrics', 'columns': ['cllr']}],
            },
        ],
    }
    validate(instance=instance, schema=generate_schema())
