from typing import Any

import confidence
import pytest

import lir
from lir import registry
from lir.config.base import GenericConfigParser
from lir.registry import _did_you_mean, _get_attribute_by_name


def test_registry_items_available():
    for name in registry.registry():
        try:
            component = registry.get(name, default_config_parser=GenericConfigParser)
            full_name = component.reference()
            _get_attribute_by_name(full_name)
        except Exception as e:
            pytest.fail(f'invalid registry entry: {name}: {e}')


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


def test_did_you_mean_suggests_close_match():
    hint = _did_you_mean('metric_bars', ['metric.metrics_bars', 'metric.cllr'])
    assert hint.startswith('; did you mean: ')
    assert "'metric.metrics_bars'" in hint


def test_did_you_mean_empty_without_close_match():
    assert _did_you_mean('totally_unrelated_key', ['metric.metrics_bars', 'metric.cllr']) == ''


def test_federated_loader_suggests_close_key():
    yaml_registry = registry.YamlRegistry(confidence.Configuration({'metric': {'metrics_bars': 'lir.registry'}}))
    loader = registry.FederatedLoader([yaml_registry, registry.ClassLoader()])

    with pytest.raises(registry.ComponentNotFoundError) as exc_info:
        loader.get('metric_bars', search_path=['metric'])
    assert "did you mean: 'metrics_bars'" in str(exc_info.value)
    assert str(exc_info.value).count('did you mean') == 1


def test_federated_loader_without_close_key_has_no_hint():
    yaml_registry = registry.YamlRegistry(confidence.Configuration({'metric': {'metrics_bars': 'lir.registry'}}))
    loader = registry.FederatedLoader([yaml_registry, registry.ClassLoader()])

    with pytest.raises(registry.ComponentNotFoundError) as exc_info:
        loader.get('definitely_not_a_real_key_xyz', search_path=['metric'])
    assert 'did you mean' not in str(exc_info.value)
