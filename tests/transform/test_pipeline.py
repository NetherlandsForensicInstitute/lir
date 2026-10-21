from pathlib import Path

import pytest

from lir.config import ConfigValue
from lir.config.transform import parse_module


@pytest.mark.parametrize(
    'module_name,config',
    [
        ('pipeline', {'steps': {'identity': 'identity'}}),
        ('logging_pipeline', {'steps': {'identity': 'identity'}}),
    ],
)
def test_parsing(module_name: str, config: dict):
    try:
        parse_module(ConfigValue.wrap([], {'method': module_name} | config), Path('/'))
    except Exception as e:
        pytest.fail(f'unable to load module `{module_name}` from registry: {e}')
