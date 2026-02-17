# test_yaml_schema.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from confidence import Configuration
from confidence.models import ConfigurationSequence
from jsonschema import ValidationError, validate


ROOT = Path(__file__).resolve().parent.parent  # Go up to project root
SCHEMA_PATH = ROOT / 'configs' / 'lir.schema.json'
YAML_DIR = ROOT / 'examples'


def _to_native_dict(cfg: Any) -> Any:
    """Recursively convert confidence Configuration objects to native Python dicts/lists.
    
    Accesses each value through cfg[key] to trigger reference resolution. The confidence
    library doesn't have a built-in method for this, so we manually traverse and resolve.
    """
    match cfg:
        case Configuration():
            return {k: _to_native_dict(cfg[k]) for k in cfg}
        case ConfigurationSequence():
            return [_to_native_dict(item) for item in cfg]
        case dict():
            return {k: _to_native_dict(v) for k, v in cfg.items()}
        case list():
            return [_to_native_dict(item) for item in cfg]
        case _:
            return cfg


def _resolve_references(data: dict) -> dict:
    """Resolve ${...} references in YAML data using confidence Library."""
    # Provide context variables needed for resolution (timestamp is always available at runtime)
    context = {'timestamp': '9999-01-01 00-00-00'}
    
    # Use confidence.Configuration to resolve ${...} references
    cfg = Configuration(data, context)
    return _to_native_dict(cfg)


@pytest.mark.parametrize('yaml_path', sorted(YAML_DIR.glob('**/*.y*ml')))
def test_yaml_files_conform_to_schema(yaml_path: Path) -> None:
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)

    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        pytest.fail(f'YAML parsing error in {yaml_path}: {e}')

    # Resolve ${...} references before validation
    try:
        data = _resolve_references(data)
    except Exception as e:
        pytest.fail(f'Reference resolution error in {yaml_path}: {e}')

    # Now validate `data` against `schema` using jsonschema
    try:
        validate(instance=data, schema=schema)
    except ValidationError as e:
        # Format the error path for better readability
        pytest.fail(f'Validation error in {yaml_path} at {e.absolute_path}: {e.message}')
