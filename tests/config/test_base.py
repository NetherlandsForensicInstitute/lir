import numbers
from numbers import Number
from pathlib import Path
from types import NoneType
from typing import Any

import numpy as np
import pytest

from lir import FeatureData
from lir.config.base import (
    ConfigValue,
    GenericConfigParser,
    GenericFunctionConfigParser,
    check_is_empty,
    config_parser,
    get_full_name,
    pop_field,
)
from lir.util import check_not_none, check_type


@pytest.mark.parametrize(
    'value_type,value',
    [
        (dict, {}),
        (list, []),
        (int, 1),
        (float, 1.1),
        (Number, 1),
        (Number, 1.1),
        (str, 'x'),
        (NoneType, None),
    ],
)
def test_wrap_unwrap(value_type: type[Any], value: Any):
    config = ConfigValue.wrap([], value)
    check_type(value_type, config.unwrap())
    assert value == config.unwrap()


@pytest.mark.parametrize(
    'config,key,value,error',
    [
        (ConfigValue.wrap([], {'key': 'value'}), 'key', 'value', None),
        (ConfigValue.wrap([], ['value1', 'value2']), 1, 'value2', None),
        (ConfigValue.wrap([], 'value'), 'key', None, ValueError),
    ],
)
def test_index_first_degree(config: ConfigValue, key: Any, value: Any, error: type[ValueError] | None) -> None:
    if error is None:
        assert config[key].unwrap() == value
    else:
        with pytest.raises(error):
            assert config[key]


def test_config_value() -> None:
    config = ConfigValue.wrap([], {'key1': 'value1', 'key2': {'key2.1': {'key2.1.1': 'value2.1.1'}}})

    assert config['key2']['key2.1']['key2.1.1'].unwrap() == 'value2.1.1'

    assert 'key1' in config
    assert 'key3' not in config
    assert list(config) == ['key1', 'key2']
    assert [x.unwrap() for x in ConfigValue.wrap([], ['value1', 'value2'])] == ['value1', 'value2']
    assert config.clone().value is not config.value
    assert config.as_dict() == config.unwrap()


def test_is_empty():
    config = ConfigValue.wrap([], {'key1': 'value1', 'key2': {'key2.1': {'key2.1.1': 'value2.1.1'}}})

    with pytest.raises(ValueError):
        check_is_empty(config)

    pop_field(config, 'key1')
    pop_field(config, 'key2')
    check_is_empty(config)


def test_pop_field():
    config = ConfigValue.wrap(
        [],
        {
            'key1': 'some text',
            'key2': 42,
            'key3': 42,
            'key4': 'some text',
            'key5': 42,
            'key6': 42,
            'key7': 42,
        },
    )

    assert pop_field(config, 'key1', validate_type=str) == 'some text'
    assert 'key1' not in config
    assert pop_field(config, 'key2', validate_type=numbers.Number) == 42
    with pytest.raises(ValueError):
        pop_field(config, 'key3', validate_type=str)
    with pytest.raises(ValueError):
        pop_field(config, 'missing_key', validate_type=str)
    assert pop_field(config, 'missing_key', validate_type=str, required=False) is None
    assert pop_field(config, 'missing_key', validate_type=str, default=42) == 42
    assert pop_field(config, 'key4', validate=lambda x: x.upper()) == 'SOME TEXT'
    assert pop_field(config, 'key5').unwrap() == 42
    assert pop_field(config, 'key6', unwrap=True) == 42
    assert pop_field(config, 'key7', validate=check_not_none) == 42


def test_get_full_name_fail():
    with pytest.raises(ValueError):
        get_full_name(FeatureData(features=np.array([1, 2])))


@pytest.mark.parametrize(
    'obj,name',
    [
        (FeatureData, 'lir.data.models.FeatureData'),
        (check_not_none, 'lir.util.check_not_none'),
    ],
)
def test_get_full_name(obj: Any, name: str):
    assert get_full_name(obj) == name


@config_parser
def my_config_parser(config: ConfigValue, output_path: Path) -> int:
    return pop_field(config, 'key', validate_type=int)


def test_config_parser():
    config = ConfigValue.wrap([], {'key': 42})
    assert my_config_parser().parse(config, Path('/')) == 42


def test_generic_config_parser():
    config = ConfigValue.wrap([], {'features': np.array([[1], [2]])})
    assert np.all(GenericConfigParser(FeatureData).parse(config, Path('/')).features == np.array([[1], [2]]))


def test_function_config_parser():
    config = ConfigValue.wrap([], {})
    assert GenericFunctionConfigParser(lambda x: x * x).parse(config, Path('/'))(3) == 9

    config = ConfigValue.wrap([], {'features': np.array([[1], [2]])})  # NB: arguments have no effect!!!
    assert GenericFunctionConfigParser(lambda x: x * x).parse(config, Path('/'))(3) == 9
