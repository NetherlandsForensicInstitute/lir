import numpy as np
import pytest

from lir import FeatureData
from lir.transform.select_instances import SelectInstances, _MatchIntPattern


@pytest.mark.parametrize(
    'pattern,elements,expected_output',
    [
        ('0', np.arange(10), np.array([0])),
        ('1-3', np.arange(10), np.array([1, 2, 3])),
        ('/2', np.arange(10), np.array([0, 2, 4, 6, 8])),
        ('3,9', np.arange(10), np.array([3, 9])),
        ('3,9,10', np.arange(10), np.array([3, 9])),
        ('/4,0', np.arange(10), np.array([0, 4, 8])),
        ('/4,1', np.arange(10), np.array([0, 1, 4, 8])),
    ],
)
def test_match_pattern(pattern: str, elements: np.ndarray, expected_output: np.ndarray):
    fn = _MatchIntPattern.parse(pattern)
    output = elements[np.vectorize(fn)(elements)]
    assert np.all(output == expected_output)

    data = FeatureData(features=elements.reshape(-1, 1))
    data = SelectInstances(fn).apply(data)
    assert np.all(data.features.reshape(-1) == expected_output)
