from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from lir.algorithms.bayeserror import ELUBBounder
from lir.algorithms.bootstraps import BootstrapAtData, BootstrapEquidistant, bootstrap
from lir.algorithms.kde import KDECalibrator
from lir.config.base import _expand
from lir.data.datasets.synthesized_normal_binary import SynthesizedNormalBinaryData, SynthesizedNormalDataClass
from lir.data.models import FeatureData, LLRData


@pytest.fixture(
    params=[
        [('kde', KDECalibrator((0.05, 0.023)))],
        [('logistic', LogisticRegression())],
        [
            ('kde', KDECalibrator((0.05, 0.023))),
            ('elub', ELUBBounder()),
        ],
        [
            ('logistic', LogisticRegression()),
            ('elub', ELUBBounder()),
        ],
        [],
    ]
)
def sample_steps_and_data(request):
    """This fixture provides sample steps and data for testing bootstrap methods.

    The steps are parameterized to test different configurations, namely KDE and LogReg,
    with and without ELUB bounding, as well as an empty step list."""
    data_spec = {
        0: SynthesizedNormalDataClass(-1, 1, 20),
        1: SynthesizedNormalDataClass(1, 1, 20),
    }
    data = SynthesizedNormalBinaryData(data_spec, seed=0)
    feature_data = data.get_instances()

    steps = request.param
    return steps, feature_data


def test_traindata_bootstrap(sample_steps_and_data):
    """Test TrainDataBootstrap with a simple logistic regression model."""
    steps, feature_data = sample_steps_and_data

    model_data = BootstrapAtData(steps).fit(feature_data)
    model_equidistant = BootstrapEquidistant(steps).fit(feature_data)

    results_data = model_data.transform(feature_data)
    results_equidistant = model_equidistant.transform(feature_data)

    assert results_data.llrs.shape == (feature_data.features.shape[0],)
    assert results_data.has_intervals
    assert results_data.llr_intervals.shape == (feature_data.features.shape[0], 2)

    # If there are no steps, skip the rest of the test.
    if not steps:
        return

    # Check that the llr values are within the interval that was calculated.
    assert np.all(results_data.llrs >= results_data.llr_intervals[:, 0])
    assert np.all(results_data.llrs <= results_data.llr_intervals[:, 1])

    assert np.all(results_equidistant.llrs >= results_equidistant.llr_intervals[:, 0])
    assert np.all(results_equidistant.llrs <= results_equidistant.llr_intervals[:, 1])


def test_interval_extrapolation(sample_steps_and_data):
    """Test that the bootstrap intervals are reasonable."""
    steps, feature_data = sample_steps_and_data

    model_equidistant = BootstrapEquidistant(steps).fit(feature_data)

    mn = np.min(feature_data.features)
    mx = np.max(feature_data.features)

    new_data = FeatureData(features=np.array([[mn - 1], [mn], [mx], [mx + 1]]))
    results: LLRData = model_equidistant.transform(new_data)
    # The difference between the interval at mn and mn-1 should be the same.
    assert np.isclose(
        results.llrs[0] - results.llr_intervals[0, 0],
        results.llrs[1] - results.llr_intervals[1, 0],
    )
    assert np.isclose(
        results.llrs[0] - results.llr_intervals[0, 1],
        results.llrs[1] - results.llr_intervals[1, 1],
    )

    # This also holds for mx and mx+1.
    assert np.isclose(
        results.llrs[2] - results.llr_intervals[2, 0],
        results.llrs[3] - results.llr_intervals[3, 0],
    )
    assert np.isclose(
        results.llrs[2] - results.llr_intervals[2, 1],
        results.llrs[3] - results.llr_intervals[3, 1],
    )


@pytest.mark.parametrize(
    'config',
    [
        (
            {
                'steps': {},
                'points': 'data',
            }
        ),
        (
            {
                'steps': {},
                'points': 'equidistant',
                'n_points': 1000,
            }
        ),
    ],
)
def test_bootstrap_config(config):
    config = _expand([], config)
    bootstrap().parse(config, Path('/'))
