from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from lir.util import (
    logodds_to_odds,
    logodds_to_probability,
    odds_to_logodds,
    odds_to_probability,
    probability_to_logodds,
    probability_to_odds,
)


def call(function: Callable, value: np.ndarray | np.number, expected: np.ndarray | np.number | type[Any]):
    if isinstance(expected, type):
        with pytest.raises(expected):
            function(value)
    else:
        actual = function(value)
        if np.isnan(expected):
            assert np.isnan(actual)
        else:
            assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    'value,expected',
    [
        (1, 0),
        (10, 1),
        (0.1, -1),
        (np.inf, np.inf),
        (0, -np.inf),
        (-1, ValueError),
        (-np.inf, ValueError),
        (np.nan, np.nan),
    ],
)
@pytest.mark.filterwarnings('error')
def test_odds_to_logodds(value: np.ndarray | np.number, expected: np.ndarray | np.number | type[Any]):
    call(odds_to_logodds, value, expected)


@pytest.mark.parametrize(
    'value,expected',
    [
        (1, 0.5),
        (10, 10 / 11),
        (0.1, 1 / 11),
        (np.inf, 1),
        (0, 0),
        (-1, ValueError),
        (-np.inf, ValueError),
        (np.nan, np.nan),
    ],
)
@pytest.mark.filterwarnings('error')
def test_odds_to_probability(value: np.ndarray | np.number, expected: np.ndarray | np.number):
    call(odds_to_probability, value, expected)


@pytest.mark.parametrize(
    'value,expected',
    [
        (1, np.inf),
        (0, -np.inf),
        (2, FloatingPointError),
        (-1, FloatingPointError),
        (np.nan, np.nan),
        (np.inf, FloatingPointError),
        (-np.inf, FloatingPointError),
        (0.5, 0),
        (0.75, np.log10(3)),
    ],
)
@pytest.mark.filterwarnings('error')
def test_probability_to_logodds(value: np.ndarray | np.number, expected: np.ndarray | np.number):
    call(probability_to_logodds, value, expected)


@pytest.mark.parametrize(
    'value,expected',
    [
        (1, np.inf),
        (0, 0),
        (2, ValueError),
        (-1, ValueError),
        (np.nan, np.nan),
        (np.inf, ValueError),
        (-np.inf, ValueError),
        (0.5, 1),
        (0.75, 3),
    ],
)
@pytest.mark.filterwarnings('error')
def test_probability_to_odds(value: np.ndarray | np.number, expected: np.ndarray | np.number):
    call(probability_to_odds, value, expected)


@pytest.mark.parametrize(
    'value,expected',
    [
        (0, 1),
        (-np.inf, 0),
        (np.inf, np.inf),
        (1, 10),
        (-1, 0.1),
        (np.nan, np.nan),
    ],
)
@pytest.mark.filterwarnings('error')
def test_logodds_to_odds(value: np.ndarray | np.number, expected: np.ndarray | np.number):
    call(logodds_to_odds, value, expected)


@pytest.mark.parametrize(
    'value,expected',
    [
        (0, 0.5),
        (-np.inf, 0),
        (np.inf, 1),
        (1, 10 / 11),
        (-1, 1 / 11),
        (np.nan, np.nan),
    ],
)
@pytest.mark.filterwarnings('error')
def test_logodds_to_probability(value: np.ndarray | np.number, expected: np.ndarray | np.number):
    call(logodds_to_probability, value, expected)
