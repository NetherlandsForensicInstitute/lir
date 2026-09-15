from typing import Any

import pytest

import lir
from lir import registry
from lir.config.base import GenericConfigParser
from lir.registry import _get_attribute_by_name, _suggest_close_matches


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


def test_suggest_close_matches_finds_close_typo():
    candidates = ['output.metrics_bars', 'output.metrics_csv', 'metric.cllr']
    assert 'output.metrics_bars' in _suggest_close_matches('metric_bars', candidates)


def test_suggest_close_matches_returns_empty_for_unrelated_key():
    candidates = ['output.metrics_bars', 'metric.cllr']
    assert _suggest_close_matches('totally_unrelated_key', candidates) == []


def test_suggest_close_matches_respects_n():
    candidates = ['aabb', 'aabc', 'aabd', 'aabe']
    matches = _suggest_close_matches('aabx', candidates, n=2)
    assert len(matches) == 2


def test_component_not_found_error_includes_suggestion():
    """The original bug: 'metric_bars' (typo) should suggest 'metrics_bars'."""
    with pytest.raises(registry.ComponentNotFoundError) as exc_info:
        registry.get('metric_bars', search_path=['output'])
    assert exc_info.value.suggestions[0] == 'metrics_bars'
    assert 'did you mean' in str(exc_info.value)


def test_component_not_found_error_without_close_match():
    with pytest.raises(registry.ComponentNotFoundError) as exc_info:
        registry.get('definitely_not_a_real_key_xyz')
    assert exc_info.value.suggestions == []
    assert 'did you mean' not in str(exc_info.value)


def test_component_not_found_error_computes_suggestions_from_candidates():
    """ComponentNotFoundError should compute suggestions itself when given key+candidates."""
    err = registry.ComponentNotFoundError(
        'no such component', key='elub', candidates=['elub_bounder', 'iv_bounder']
    )
    assert err.suggestions == ['elub_bounder']
    assert 'did you mean' in str(err)


def test_component_not_found_error_without_candidates_has_no_suggestions():
    err = registry.ComponentNotFoundError('no such component')
    assert err.suggestions == []
    assert 'did you mean' not in str(err)


def test_suggest_close_matches_finds_prefix_abbreviation():
    """User typed an abbreviation; suggest the candidate that starts with that prefix."""
    assert _suggest_close_matches('elub', ['elub_bounder', 'iv_bounder']) == ['elub_bounder']


def test_suggest_close_matches_ignores_short_prefix():
    """A 1- or 2-char key should not match everything via prefix."""
    candidates = ['elub_bounder', 'iv_bounder']
    assert _suggest_close_matches('e', candidates) == []
