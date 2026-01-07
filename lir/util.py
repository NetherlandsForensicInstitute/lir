import collections
import inspect
import warnings
from functools import partial
from typing import Any, TypeVar

import numpy as np


LR = collections.namedtuple('LR', ['lr', 'p0', 'p1'])


AnyType = TypeVar('AnyType', bound=Any)


def check_type(type_class: type[AnyType], v: Any, message: str | None = None) -> AnyType:
    if isinstance(v, type_class):
        return v
    else:
        message = message or f'expected type: {type_class}'
        raise ValueError(f'{message}; found: {type(v)}')


def get_classes_from_Xy(X: np.ndarray, y: np.ndarray, classes: list[Any] | None = None) -> np.ndarray:
    assert len(X.shape) >= 1, f'expected: X has at least 1 dimensions; found: {len(X.shape)} dimensions'
    assert len(y.shape) == 1, f'expected: y is a 1-dimensional array; found: {len(y.shape)} dimensions'
    assert X.shape[0] == y.size, f'dimensions of X and y do not match; found: {X.shape[0]} != {y.size}'

    return np.unique(y) if classes is None else np.asarray(classes)


def Xn_to_Xy(*Xn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Xn to Xy format.

    Xn is a format where samples are divided into separate variables based on class.
    Xy is a format where all samples are concatenated, with an equal length variable y indicating class.
    """
    split_Xn = [np.asarray(X) for X in Xn]
    X = np.concatenate(split_Xn)
    y = np.concatenate([np.ones((X.shape[0],), dtype=np.int8) * i for i, X in enumerate(split_Xn)])
    return X, y


def Xy_to_Xn(X: np.ndarray, y: np.ndarray, classes: list[int] | None = None) -> list[np.ndarray]:
    """
    Convert Xy to Xn format.

    Xn is a format where samples are divided into separate variables based on class.
    Xy is a format where all samples are concatenated, with an equal length variable y indicating class.
    """

    if classes is None:
        classes = [0, 1]

    new_classes = get_classes_from_Xy(X, y, classes)
    return [X[y == yvalue] for yvalue in new_classes]


FloatOrArray = TypeVar('FloatOrArray', np.ndarray, float)


def odds_to_probability(odds: FloatOrArray) -> FloatOrArray:
    """
    Converts odds to a probability

    Returns
    -------
       1                , for odds values of inf
       odds / (1 + odds), otherwise
    """
    inf_values = odds == np.inf
    with np.errstate(invalid='ignore'):
        p = np.divide(odds, (1 + odds))
    p[inf_values] = 1
    return p


def probability_to_odds(p: FloatOrArray) -> FloatOrArray:
    """
    Converts a probability to odds
    """
    with np.errstate(divide='ignore'):
        return p / (1 - p)


def probability_to_logodds(p: FloatOrArray) -> FloatOrArray:
    """
    Converts probability values to their log odds with base 10.
    """
    with np.errstate(divide='ignore'):
        complement = 1 - p
        return np.log10(p) - np.log10(complement)


def logodds_to_probability(log_odds: FloatOrArray) -> FloatOrArray:
    return odds_to_probability(logodds_to_odds(log_odds))


def logodds_to_odds(log_odds: FloatOrArray) -> FloatOrArray:
    with np.errstate(divide='ignore'):
        return 10**log_odds


def odds_to_logodds(odds: FloatOrArray) -> FloatOrArray:
    return np.log10(odds)


def ln_to_log10(ln_data: FloatOrArray) -> FloatOrArray:
    return np.log10(np.e) * ln_data


def warn_deprecated() -> None:
    warnings.warn(
        f'the function `{inspect.stack()[1].function}` is no longer maintained; '
        'please check documentation for alternatives',
        stacklevel=2,
    )


class Bind(partial):
    """
    An improved version of partial which accepts Ellipsis (...) as a placeholder.
    Can be used to fix parameters not at the end of the list of parameters (which is a limitation of partial).
    """

    def __call__(self, *args: Any, **keywords: Any) -> Any:
        keywords = {**self.keywords, **keywords}
        iargs = iter(args)
        args = tuple(next(iargs) if arg is ... else arg for arg in self.args)
        return self.func(*args, *iargs, **keywords)


def check_misleading_finite(values: np.ndarray, labels: np.ndarray) -> None:
    """
    Check whether all values are either finite or not misleading.
    """

    # give error message if H1's contain zeros and H2's contain ones
    if np.any(np.isneginf(values[labels == 1])) and np.any(np.isposinf(values[labels == 0])):
        raise ValueError('invalid input: -inf found for H1 and inf found for H2')
    # give error message if H1's contain zeros
    if np.any(np.isneginf(values[labels == 1])):
        raise ValueError('invalid input: -inf found for H1')
    # give error message if H2's contain ones
    if np.any(np.isposinf(values[labels == 0])):
        raise ValueError('invalid input: inf found for H2')
