from pathlib import Path

import pytest

from lir.config import ConfigValue, YamlParseError
from lir.config.transform import GenericTransformerConfigParser
from lir.config.util import parse_config
from lir.transform import Identity


def test_parse_config_non_mapping_section_mentions_yaml_path():
    config = ConfigValue.wrap(['splits'], [1, 2])
    with pytest.raises(
        YamlParseError,
        match=r'splits: invalid configuration section; expected a mapping with a `strategy` field',
    ):
        parse_config(config, Path(), method_key='strategy', search_path=['data_strategies'])


def test_parse_config_invalid_method_value_mentions_yaml_path():
    config = ConfigValue.wrap(['splits'], {'strategy': 1})
    with pytest.raises(YamlParseError, match='splits: illegal value for field `strategy`'):
        parse_config(config, Path(), method_key='strategy', search_path=['data_strategies'])


def test_parse_config_shorthand_form():
    config = ConfigValue.wrap(['modules'], 'identity')
    module = parse_config(
        config,
        Path(),
        method_key='method',
        allow_shorthand=True,
        default_config_parser=GenericTransformerConfigParser,
        search_path=['modules'],
    )
    assert isinstance(module, Identity)


def test_parse_config_unknown_component_mentions_name():
    config = ConfigValue.wrap(['modules'], {'method': 'no_such_module'})
    with pytest.raises(
        YamlParseError,
        match='modules: component not found: no_such_module',
    ):
        parse_config(
            config,
            Path(),
            method_key='method',
            default_config_parser=GenericTransformerConfigParser,
            search_path=['modules'],
        )
