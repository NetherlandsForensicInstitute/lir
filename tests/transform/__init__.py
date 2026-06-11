from functools import partial

import numpy as np
import pytest

from lir import FeatureData
from lir.transform import FunctionTransformer
from lir.util import check_type


@pytest.mark.parametrize(
    'features,transformer,expected_result',
    [
        (
            np.array([1, 10, 100]),
            FunctionTransformer(np.log10),
            np.array([0, 1, 2]),
        ),
        (
            np.arange(5),
            FunctionTransformer(partial(np.clip, min=2)),
            np.array([2, 2, 2, 3, 4]),
        ),
    ],
)
def test_function_transform(features: np.ndarray, transformer: FunctionTransformer, expected_result: np.ndarray):
    data = transformer.fit_apply(FeatureData(features=features))
    data = check_type(FeatureData, data)
    assert np.all(data.features.reshape(-1) == expected_result)
