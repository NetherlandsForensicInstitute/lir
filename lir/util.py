import collections
import datetime
import inspect
import json
import warnings
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import yaml
from confidence import Configuration
from confidence.models import ConfigurationSequence
from jsonschema import validate


LR = collections.namedtuple('LR', ['lr', 'p0', 'p1'])


AnyType = TypeVar('AnyType', bound=Any)


def check_type[AnyType: Any](type_class: type[AnyType], v: Any, message: str | None = None) -> AnyType:
    """Check if a given input is of the expected, specified type."""
    if isinstance(v, type_class):
        return v
    else:
        message = message or f'expected type: {type_class}'
        raise ValueError(f'{message}; found: {type(v)}')


def get_classes_from_Xy(X: np.ndarray, y: np.ndarray, classes: list[Any] | None = None) -> np.ndarray:
    """Get the classification classes from labeled data."""
    assert len(X.shape) >= 1, f'expected: X has at least 1 dimensions; found: {len(X.shape)} dimensions'
    assert len(y.shape) == 1, f'expected: y is a 1-dimensional array; found: {len(y.shape)} dimensions'
    assert X.shape[0] == y.size, f'dimensions of X and y do not match; found: {X.shape[0]} != {y.size}'

    return np.unique(y) if classes is None else np.asarray(classes)


def Xn_to_Xy(*Xn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert Xn to Xy format.

    Xn is a format where samples are divided into separate variables based on class.
    Xy is a format where all samples are concatenated, with an equal length variable y indicating class.
    """
    split_Xn = [np.asarray(X) for X in Xn]
    X = np.concatenate(split_Xn)
    y = np.concatenate([np.ones((X.shape[0],), dtype=np.int8) * i for i, X in enumerate(split_Xn)])
    return X, y


def Xy_to_Xn(X: np.ndarray, y: np.ndarray, classes: list[int] | None = None) -> list[np.ndarray]:
    """Convert Xy to Xn format.

    Xn is a format where samples are divided into separate variables based on class.
    Xy is a format where all samples are concatenated, with an equal length variable y indicating class.
    """
    if classes is None:
        classes = [0, 1]

    new_classes = get_classes_from_Xy(X, y, classes)
    return [X[y == yvalue] for yvalue in new_classes]


FloatOrArray = TypeVar('FloatOrArray', np.ndarray, float)


def odds_to_probability[FloatOrArray: (np.ndarray, float)](odds: FloatOrArray) -> FloatOrArray:
    """Converts odds to a probability.

    Returns:
    - 1                , for odds values of inf
    - odds / (1 + odds), otherwise
    """
    inf_values = odds == np.inf
    with np.errstate(invalid='ignore'):
        p = np.divide(odds, (1 + odds))
    p[inf_values] = 1
    return p


def probability_to_odds[FloatOrArray: (np.ndarray, float)](p: FloatOrArray) -> FloatOrArray:
    """Converts a probability to odds."""
    with np.errstate(divide='ignore'):
        return p / (1 - p)


def probability_to_logodds[FloatOrArray: (np.ndarray, float)](p: FloatOrArray) -> FloatOrArray:
    """Converts probability values to their log odds with base 10."""
    with np.errstate(divide='ignore'):
        complement = 1 - p
        return np.log10(p) - np.log10(complement)


def logodds_to_probability[FloatOrArray: (np.ndarray, float)](log_odds: FloatOrArray) -> FloatOrArray:
    """Convert 10-base logarithm of odds to probability."""
    return odds_to_probability(logodds_to_odds(log_odds))


def logodds_to_odds[FloatOrArray: (np.ndarray, float)](log_odds: FloatOrArray) -> FloatOrArray:
    """Convert 10-base logarithm odds to odds."""
    with np.errstate(divide='ignore'):
        return 10**log_odds


def odds_to_logodds[FloatOrArray: (np.ndarray, float)](odds: FloatOrArray) -> FloatOrArray:
    """Convert odds to 10-base logarithm odds."""
    return np.log10(odds)


def ln_to_log10[FloatOrArray: (np.ndarray, float)](ln_data: FloatOrArray) -> FloatOrArray:
    """Convert natural logarithm to 10-base logarithm."""
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

    Accesses each value through cfg[key] to trigger reference resolution. The confidence
    library doesn't have a built-in method for this, so we manually traverse and resolve.
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
    schema_path = Path(__file__).parent.parent / 'configs' / 'lir.schema.json'

    if not schema_path.exists():
        raise FileNotFoundError(f'Schema file not found: {schema_path}')

    if not yaml_path.exists():
        raise FileNotFoundError(f'YAML file not found: {yaml_path}')

    with open(schema_path) as f:
        schema = json.load(f)

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Resolve ${...} references before validation
    context = {'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}  # noqa: DTZ005
    cfg = Configuration(data, context)
    data = to_native_dict(cfg)

    # Validate data against schema
    validate(instance=data, schema=schema)


class Bind(partial):
    """Wrap `partial` to support the ellipsis (...) as a placeholder.

    Can be used to fix parameters not at the end of the list of parameters (which is a limitation of partial).
    """

    def __call__(self, *args: Any, **keywords: Any) -> Any:
        """Extends `partial` and accepts the ellipsis as a placeholder."""
        keywords = {**self.keywords, **keywords}
        iargs = iter(args)
        args = tuple(next(iargs) if arg is ... else arg for arg in self.args)
        return self.func(*args, *iargs, **keywords)
