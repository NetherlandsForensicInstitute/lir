# test_yaml_schema.py
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import ValidationError, validate


ROOT = Path(__file__).resolve().parent.parent  # Go up to project root
SCHEMA_PATH = ROOT / 'configs' / 'lir.schema.json'
YAML_DIR = ROOT / 'examples'


@pytest.mark.parametrize('yaml_path', sorted(YAML_DIR.glob('**/*.y*ml')))
def test_yaml_files_conform_to_schema(yaml_path: Path) -> None:
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)

    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        pytest.fail(f'YAML parsing error in {yaml_path}: {e}')

    # Now validate `data` against `schema` using jsonschema
    try:
        validate(instance=data, schema=schema)
    except ValidationError as e:
        # Format the error path for better readability
        pytest.fail(f'Validation error in {yaml_path} at {e.absolute_path}: {e.message}')
