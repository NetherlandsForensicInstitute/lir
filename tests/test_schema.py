# test_yaml_schema.py
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from jsonschema import ValidationError

from lir.util import validate_yaml


ROOT = Path(__file__).resolve().parent.parent  # Go up to project root
SCHEMA_PATH = ROOT / 'configs' / 'lir.schema.json'
YAML_DIR = ROOT / 'examples'


@pytest.mark.parametrize('yaml_path', sorted(YAML_DIR.glob('**/*.y*ml')))
def test_yaml_files_conform_to_schema(yaml_path: Path) -> None:
    """Test that all YAML files in the examples directory conform to the schema."""
    try:
        validate_yaml(yaml_path)
    except (FileNotFoundError, yaml.YAMLError, ValidationError) as e:
        pytest.fail(f'Validation failed for {yaml_path}: {e}')
