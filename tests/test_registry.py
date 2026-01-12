from typing import Any

import pytest

import lir
from lir import registry
from lir.config.base import GenericConfigParser


def test_registry_items_available():
    for name in registry.registry():
        registry.get(name, default_config_parser=GenericConfigParser)


@pytest.mark.parametrize(
    'obj_name,expected',
    [
        pytest.param('lir', lir),
        pytest.param('lir.registry', lir.registry),
        pytest.param('lir.registry.ConfigParserLoader', lir.registry.ConfigParserLoader),
        pytest.param(
            'lir.registry.ConfigParserLoader._get_config_parser',
            lir.registry.ConfigParserLoader._get_config_parser,
        ),
    ],
)
def test_get_attribute_by_name(obj_name: str, expected: Any):
    obj = lir.registry._get_attribute_by_name(obj_name)
    assert obj is not None, f'no result for {obj_name}'
    assert obj == expected
