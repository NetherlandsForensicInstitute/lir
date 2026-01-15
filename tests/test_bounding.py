import numpy as np
import pytest

from lir.bounding import NSourceBounder, StaticBounder
from lir.data.models import LLRData


@pytest.mark.parametrize(
    'lower_bound,upper_bound,llrs,expected_result',
    [
        (
            -1,
            1,
            np.arange(3),
            np.array([0, 1, 1]),
        ),
        (
            0,
            0,
            np.arange(3),
            np.array([0, 0, 0]),
        ),
        (
            10,
            10,
            np.arange(3),
            np.array([10, 10, 10]),
        ),
        (
            -1,
            1,
            np.arange(5) - 2,
            np.array([-1, -1, 0, 1, 1]),
        ),
        (
            -1.0,
            1.0,
            np.arange(5) - 2,
            np.array([-1, -1, 0, 1, 1]),
        ),
        (
            None,
            1,
            np.arange(5) - 2,
            np.array([-2, -1, 0, 1, 1]),
        ),
        (
            -1,
            None,
            np.arange(5) - 2,
            np.array([-1, -1, 0, 1, 2]),
        ),
        (
            None,
            None,
            np.arange(5) - 2,
            np.array([-2, -1, 0, 1, 2]),
        ),
    ],
)
def test_static_bounder(
    lower_bound: float,
    upper_bound: float,
    llrs: np.ndarray,
    expected_result: np.ndarray,
):
    bounder = StaticBounder(lower_bound, upper_bound)
    labels = np.concatenate([np.zeros(1), np.ones(llrs.shape[0] - 1)])
    llrs = LLRData(features=llrs, labels=labels)
    assert np.all(expected_result == bounder.apply(llrs).llrs)
    assert np.all(expected_result == bounder.fit_apply(llrs).llrs)
    assert np.all(expected_result == bounder.fit(llrs).apply(llrs).llrs)


@pytest.mark.parametrize(
    'lower_bound,upper_bound,llrs,labels',
    [
        (  # no labels argument
            -1,
            1,
            np.arange(3),
            None,
        ),
        (  # too few labels
            -1,
            1,
            np.arange(3),
            np.array([1, 2]),
        ),
        (  # labels other than 0, 1
            -1,
            1,
            np.arange(3),
            np.arange(3),
        ),
        (  # labels other than 0, 1
            -1,
            1,
            np.arange(3),
            np.array([np.nan, np.nan, np.nan]),
        ),
        (  # too many labels
            -1,
            1,
            np.arange(3),
            np.array([0, 1, 1, 1]),
        ),
        (  # labels other than 0, 1
            -1,
            1,
            np.arange(3),
            np.array([0.00000001, 1, 1]),
        ),
        (  # invalid dimensions for llrs
            -1,
            1,
            np.zeros((3, 1)),
            np.array([0, 1, 1]),
        ),
        (  # invalid dimensions for labels
            -1,
            1,
            np.zeros(3),
            np.array([[0, 1, 1]]),
        ),
        (  # invalid bounds
            2,
            1,
            np.zeros(3),
            np.array([[0, 1, 1]]),
        ),
    ],
)
def test_static_bounder_illegal_input(lower_bound: float, upper_bound: float, llrs: np.ndarray, labels: np.ndarray):
    bounder = StaticBounder(lower_bound, upper_bound)
    with pytest.raises(TypeError):
        bounder.fit(llrs, labels).apply(llrs)


def test_n_source_bounder():
    bounder = NSourceBounder()
    llrs = LLRData(
        features=np.array([0.5, -0.2, 1.0, 0.3, -0.7, 0.8]),
        labels=np.array([1, 0, 1, 0, 0, 1]),
        source_ids=np.array(['A', 'A', 'B', 'B', 'C', 'C']),
    )
    expected_lower_bound = -np.log10(3)
    expected_upper_bound = np.log10(3)

    applied = bounder.fit_apply(llrs)
    assert applied.llr_lower_bound == expected_lower_bound
    assert applied.llr_upper_bound == expected_upper_bound
    assert np.allclose(
        applied.llrs,
        np.clip(
            llrs.features,
            expected_lower_bound,
            expected_upper_bound,
        ).flatten(),
    )
