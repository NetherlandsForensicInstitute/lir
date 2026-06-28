from numbers import Number
from types import NoneType
from typing import Any

import pytest

from lir.config.base import ConfigValue


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
def test_check_type_good(value_type: type[Any], value: Any):
    config = ConfigValue.wrap([], value)
    config.check_type(value_type)


@pytest.mark.parametrize(
    'value_type,value',
    [
        (dict, 1),
        (list, 1),
        (int, 1.1),
        (float, 1),
        (Number, '1'),
        (str, 1),
        (NoneType, 1),
    ],
)
def test_check_type_bad(value_type: type[Any], value: Any):
    config = ConfigValue.wrap([], value)
    with pytest.raises(ValueError):
        config.check_type(value_type)
        pytest.fail(f'{value} of type {type(value)} should be rejected because it is not an instance of {value_type}')
