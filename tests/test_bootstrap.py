from sklearn.linear_model import LogisticRegression
from lir.data.datasets.synthesized_normal_binary import SynthesizedNormalBinaryData, SynthesizedNormalDataClass
from lir.lrsystems.bootstraps import BootstrapAtData
import numpy as np


def test_traindata_bootstrap():
    """Test TrainDataBootstrap with a simple logistic regression model."""
    data_spec = {
        0: SynthesizedNormalDataClass(-1, 1, 10),
        1: SynthesizedNormalDataClass(1, 1, 10),
    }
    data = SynthesizedNormalBinaryData(data_spec, seed=0)
    feature_data = data.get_instances()

    steps = [
        ("logreg", LogisticRegression(solver="lbfgs")),
    ]

    results = BootstrapAtData(steps).fit(feature_data).transform(feature_data)

    # Check that the llr values are within the inteval it has calculated.
    assert np.all(results.llrs > results.llr_intervals[:, 0])
    assert np.all(results.llrs < results.llr_intervals[:, 1])


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
