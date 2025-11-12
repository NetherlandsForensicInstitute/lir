from pathlib import Path

from sklearn.linear_model import LogisticRegression

from lir.config.base import _expand
from lir.data.datasets.synthesized_normal_binary import SynthesizedNormalBinaryData, SynthesizedNormalDataClass
from lir.algorithms.bootstraps import BootstrapAtData, BootstrapEquidistant, bootstrap
import numpy as np
import pytest
from lir.data.models import FeatureData, LLRData


def test_traindata_bootstrap(sample_steps_and_data):
    """Test TrainDataBootstrap with a simple logistic regression model."""
    steps, feature_data = sample_steps_and_data

    model_data = BootstrapAtData(steps).fit(feature_data)
    model_equidistant = BootstrapEquidistant(steps).fit(feature_data)

    results_data = model_data.transform(feature_data)
    results_equidistant = model_equidistant.transform(feature_data)

    # Check that the llr values are within the inteval it has calculated.
    assert np.all(results_data.llrs > results_data.llr_intervals[:, 0])
    assert np.all(results_data.llrs < results_data.llr_intervals[:, 1])

    assert np.all(results_equidistant.llrs > results_equidistant.llr_intervals[:, 0])
    assert np.all(results_equidistant.llrs < results_equidistant.llr_intervals[:, 1])


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


@pytest.fixture
def sample_steps_and_data():
    data_spec = {
        0: SynthesizedNormalDataClass(-1, 1, 10),
        1: SynthesizedNormalDataClass(1, 1, 10),
    }
    data = SynthesizedNormalBinaryData(data_spec, seed=0)
    feature_data = data.get_instances()

    steps = [
        ('logreg', LogisticRegression(solver='lbfgs')),
    ]
    return steps, feature_data


def test_traindata_bootstrap_empty_pipeline():
    """Test TrainDataBootstrap with an empty pipeline."""
    data_spec = {
        0: SynthesizedNormalDataClass(0, 0, 10),
    }
    data = SynthesizedNormalBinaryData(data_spec, seed=0)
    feature_data = data.get_instances()

    steps = []
    results = BootstrapAtData(steps).fit(feature_data).transform(feature_data)

    # This is the most simple system possible, where all features and bounds should be zero.
    assert np.all(results.features == 0)


@pytest.mark.parametrize("config", [
    (
            {
                "steps": None,
                "points": "data",
            }
    ),
    (
            {
                "steps": None,
                "points": "equidistant",
                "n_points": 1000,
            }
    ),
])
def test_bootstrap_config(config):
    config = _expand([], config)
    bootstrap().parse(config, Path("/"))
