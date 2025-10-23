import numpy as np
import pytest

from lir.bounding import StaticBounder


@pytest.mark.parametrize(
    "lower_bound,upper_bound,llrs,expected_result",
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
    assert np.all(expected_result == bounder.transform(llrs))
    assert np.all(expected_result == bounder.fit(llrs, labels).transform(llrs))


@pytest.mark.parametrize(
    "lower_bound,upper_bound,llrs,labels",
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
def test_static_bounder_illegal_input(
    lower_bound: float, upper_bound: float, llrs: np.ndarray, labels: np.ndarray
):
    bounder = StaticBounder(lower_bound, upper_bound)
    with pytest.raises(Exception):
        bounder.fit(llrs, labels).transform(llrs)
