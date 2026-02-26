import collections
import datetime
import inspect
import json
import warnings
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from confidence import Configuration, loadf
from confidence.models import ConfigurationSequence
from jsonschema import validate


LR = collections.namedtuple('LR', ['lr', 'p0', 'p1'])


AnyType = TypeVar('AnyType', bound=Any)


def check_type[AnyType: Any](type_class: type[AnyType], v: Any, message: str | None = None) -> AnyType:
    """
    Check if a given input is of the expected, specified type. If so, return the input value.

    Parameters
    ----------
    type_class : type
        The expected type of the input value.
    v : Any
        The input value to be checked against the expected type.
    message : str, optional
        An optional message to be included in the error if the type check fails. If not provided, a default message
        indicating the expected type will be used.

    Returns
    -------
    AnyType
        The input value `v` if it is of the expected type.
    """
    if isinstance(v, type_class):
        return v
    else:
        message = message or f'expected type: {type_class}'
        raise ValueError(f'{message}; found: {type(v)}')


def get_classes_from_Xy(X: np.ndarray, y: np.ndarray, classes: list[Any] | None = None) -> np.ndarray:
    """
    Get the classification classes from labeled data.

    Parameters
    ----------
    X : np.ndarray
        The input data array, where rows correspond to samples and columns correspond to features.
    y : np.ndarray
        The target labels corresponding to each sample in `X`. This should be a 1-dimensional array.
    classes : list[Any] | None, optional
        An optional list of classes to be used. If not provided, the unique values in `y` will be used.

    Returns
    -------
    np.ndarray
        An array of unique classes found in `y` if `classes` is None; otherwise, an array of the provided `classes`.
    """
    assert len(X.shape) >= 1, f'expected: X has at least 1 dimensions; found: {len(X.shape)} dimensions'
    assert len(y.shape) == 1, f'expected: y is a 1-dimensional array; found: {len(y.shape)} dimensions'
    assert X.shape[0] == y.size, f'dimensions of X and y do not match; found: {X.shape[0]} != {y.size}'

    return np.unique(y) if classes is None else np.asarray(classes)


def Xn_to_Xy(*Xn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Xn to Xy format.

    Parameters
    ----------
    *Xn : np.ndarray
        Variable number of arrays, where each array corresponds to a class and contains the samples for that class.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - X: A 2D array where all samples from the input arrays are concatenated together.
        - y: A 1D array of the same length as the number of samples in X, where each element indicates the class label.
    """
    split_Xn = [np.asarray(X) for X in Xn]
    X = np.concatenate(split_Xn)
    y = np.concatenate([np.ones((X.shape[0],), dtype=np.int8) * i for i, X in enumerate(split_Xn)])
    return X, y


def Xy_to_Xn(X: np.ndarray, y: np.ndarray, classes: list[int] | None = None) -> list[np.ndarray]:
    """
    Convert Xy to Xn format.

    Parameters
    ----------
    X : np.ndarray
        A 2D array where rows correspond to samples and columns correspond to features.
    y : np.ndarray
        A 1D array of the same length as the number of samples in X, where each element indicates the class label.
    classes : list[int] | None, optional
        An optional list of class labels to be used for splitting the data. If not provided, the unique values in `y`
        will be used as class labels.

    Returns
    -------
    list[np.ndarray]
        A list of arrays, where each array corresponds to a class and contains the samples for that class.
    """
    if classes is None:
        classes = [0, 1]

    new_classes = get_classes_from_Xy(X, y, classes)
    return [X[y == yvalue] for yvalue in new_classes]


FloatOrArray = TypeVar('FloatOrArray', np.ndarray, float)


def odds_to_probability[FloatOrArray: (np.ndarray, float)](odds: FloatOrArray) -> FloatOrArray:
    """
    Convert odds to a probability.

    Parameters
    ----------
    odds : FloatOrArray
        The odds to be converted to probability.

    Returns
    -------
    FloatOrArray
        The input odds converted to probability. This is 1 if the input odds is infinity, and otherwise calculated as
        odds / (1 + odds).
    """
    inf_values = odds == np.inf
    with np.errstate(invalid='ignore'):
        p = np.divide(odds, (1 + odds))
    p[inf_values] = 1
    return p


def probability_to_odds[FloatOrArray: (np.ndarray, float)](p: FloatOrArray) -> FloatOrArray:
    """
    Convert a probability to odds.

    Parameters
    ----------
    p : FloatOrArray
        The probability to be converted to odds.

    Returns
    -------
    FloatOrArray
        The input probability converted to odds. This is infinity if the input probability is 1, and otherwise
        calculated as p / (1 - p).
    """
    with np.errstate(divide='ignore'):
        return p / (1 - p)


def probability_to_logodds[FloatOrArray: (np.ndarray, float)](p: FloatOrArray) -> FloatOrArray:
    """
    Convert probability values to their log odds with base 10.

    Parameters
    ----------
    p : FloatOrArray
        The probability values to be converted to log odds.

    Returns
    -------
    FloatOrArray
        The input probability values converted to log odds with base 10.
    """
    with np.errstate(divide='ignore'):
        complement = 1 - p
        return np.log10(p) - np.log10(complement)


def logodds_to_probability[FloatOrArray: (np.ndarray, float)](log_odds: FloatOrArray) -> FloatOrArray:
    """
    Convert 10-base logarithm of odds to probability.

    Parameters
    ----------
    log_odds : FloatOrArray
        The 10-base logarithm of odds to be converted to probability.

    Returns
    -------
    FloatOrArray
        The input 10-base logarithm of odds converted to probability.
    """
    return odds_to_probability(logodds_to_odds(log_odds))


def logodds_to_odds[FloatOrArray: (np.ndarray, float)](log_odds: FloatOrArray) -> FloatOrArray:
    """
    Convert 10-base logarithm odds to odds.

    Parameters
    ----------
    log_odds : FloatOrArray
        The 10-base logarithm of odds to be converted to odds.

    Returns
    -------
    FloatOrArray
        The input 10-base logarithm of odds converted to odds.
    """
    with np.errstate(divide='ignore'):
        return 10**log_odds


def odds_to_logodds[FloatOrArray: (np.ndarray, float)](odds: FloatOrArray) -> FloatOrArray:
    """
    Convert odds to 10-base logarithm odds.

    Parameters
    ----------
    odds : FloatOrArray
        The odds to be converted to 10-base logarithm odds.

    Returns
    -------
    FloatOrArray
        The input odds converted to 10-base logarithm odds.
    """
    return np.log10(odds)


def ln_to_log10[FloatOrArray: (np.ndarray, float)](ln_data: FloatOrArray) -> FloatOrArray:
    """
    Convert natural logarithm to 10-base logarithm.

    Parameters
    ----------
    ln_data : FloatOrArray
        Data in natural logarithm form to be converted to 10-base logarithm.

    Returns
    -------
    FloatOrArray
        The input data converted from natural logarithm to 10-base logarithm.
    """
    return np.log10(np.e) * ln_data


def warn_deprecated() -> None:
    """Provide template message for deprecated functions."""
    warnings.warn(
        f'the function `{inspect.stack()[1].function}` is no longer maintained; '
        'please check documentation for alternatives',
        stacklevel=2,
    )


def to_native_dict(cfg: Any) -> Any:
    """Recursively convert confidence Configuration objects to native Python dicts/lists.

    Accesses each value through cfg[key] to trigger reference resolution. The confidence ibrary doesn't have a built-in
    method for this, so we manually traverse and resolve.

    Similary to lir.config.base._expand, but this method returns native dicts/lists instead of
    ContextAwareDict/ContextAwareList.
    """
    match cfg:
        case Configuration():
            return {k: to_native_dict(cfg[k]) for k in cfg}
        case ConfigurationSequence():
            return [to_native_dict(item) for item in cfg]
        case dict():
            return {k: to_native_dict(v) for k, v in cfg.items()}
        case list():
            return [to_native_dict(item) for item in cfg]
        case _:
            return cfg


def validate_yaml(yaml_path: Path) -> None:
    """Validate a YAML file against the schema.

    :param yaml_path: path to the YAML file to validate
    :raises FileNotFoundError: if the YAML or schema file doesn't exist
    :raises yaml.YAMLError: if the YAML is invalid
    :raises ValidationError: if the YAML doesn't conform to the schema
    """
    schema_path = Path(__file__).parent.parent / 'lir.schema.json'

    if not schema_path.exists():
        raise FileNotFoundError(f'Schema file not found: {schema_path}')

    if not yaml_path.exists():
        raise FileNotFoundError(f'YAML file not found: {yaml_path}')

    with open(schema_path) as f:
        schema = json.load(f)

    # Resolve ${...} references before validation
    context = {'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}  # noqa: DTZ005
    cfg = Configuration(loadf(yaml_path), context)
    data = to_native_dict(cfg)

    # Validate data against schema
    validate(instance=data, schema=schema)


class Bind(partial):
    """
    Wrap `partial` to support the ellipsis (...) as a placeholder.

    Can be used to fix parameters not at the end of the list of parameters (which is a limitation of partial).
    """

    def __call__(self, *args: Any, **keywords: Any) -> Any:
        """
        Extend `partial` and accept the ellipsis as a placeholder.

        Parameters
        ----------
        *args
            Positional arguments to be passed to the original function.
        **keywords
            Keyword arguments to be passed to the original function.

        Returns
        -------
        Any
            A new function with the same behavior as the original function, but with the specified parameters fixed.
        """
        keywords = {**self.keywords, **keywords}
        iargs = iter(args)
        args = tuple(next(iargs) if arg is ... else arg for arg in self.args)
        return self.func(*args, *iargs, **keywords)
