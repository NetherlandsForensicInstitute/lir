from numbers import Number
from types import NoneType
from typing import Any

import pytest

from lir.config.base import AnyType, ConfigValue
from lir.util import check_type


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
def test_wrap_unwrap(value_type: type[AnyType], value: Any):
    config = ConfigValue.wrap([], value)
    check_type(value_type, config.unwrap())
    assert value == config.unwrap()
