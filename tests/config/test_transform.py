from pathlib import Path

import numpy as np
import pytest

from lir import FeatureData
from lir.config.base import ContextAwareDict
from lir.config.transform import GenericTransformerConfigParser
from lir.util import check_type


@pytest.mark.parametrize(
    'obj,kwargs,features,expected_result',
    [
        (
            np.log10,
            {},
            np.array([1, 10, 100]),
            np.array([0, 1, 2]),
        ),
        (
            np.clip,
            {'min': 2},
            np.arange(5),
            np.array([2, 2, 2, 3, 4]),
        ),
    ],
)
def test_generic_transformer_config_parser(
    obj: object, kwargs: dict[str, str], features: np.ndarray, expected_result: np.ndarray
):
    transformer = GenericTransformerConfigParser(obj).parse(ContextAwareDict([], kwargs), Path())
    data = transformer.fit_apply(FeatureData(features=features))
    data = check_type(FeatureData, data)
    assert np.all(data.features.reshape(-1) == expected_result)
