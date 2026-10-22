from pathlib import Path

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from lir.algorithms.svm import CalibratedSVC
from lir.config.base import ConfigValue
from lir.config.transform import parse_module
from lir.data.models import FeatureData


@pytest.fixture
def instances() -> FeatureData:
    rng = np.random.default_rng(42)
    features = np.concatenate(
        [
            rng.normal(loc=-1, scale=1, size=(100, 2)),
            rng.normal(loc=1, scale=1, size=(100, 2)),
        ]
    )
    hypothesis = np.concatenate([np.zeros(100), np.ones(100)])
    return FeatureData(features=features, hypothesis=hypothesis)


def test_calibrated_svc_wraps_calibrated_classifier_cv():
    clf = CalibratedSVC(kernel='linear')

    # the estimator should be a calibrated classifier around an SVM
    assert isinstance(clf.estimator, CalibratedClassifierCV)
    assert isinstance(clf.estimator.estimator, SVC)
    assert clf.estimator.ensemble is False


def test_calibrated_svc_fit_apply(instances: FeatureData):
    fitted = CalibratedSVC(kernel='linear').fit(instances)
    probabilities = fitted.apply(instances).features

    # probabilities should be in (0, 1) and separate the classes
    assert probabilities.shape == (200, 1)
    assert np.all(probabilities > 0)
    assert np.all(probabilities < 1)
    # the first 100 instances have hypothesis 0, so their probability of hypothesis 1 should be lower
    assert np.mean(probabilities[:100]) < np.mean(probabilities[100:])


def test_calibrated_svc_probability_argument_is_deprecated(instances: FeatureData):
    with pytest.warns(DeprecationWarning, match='`probability` is deprecated'):
        clf = CalibratedSVC(probability=True, kernel='linear')

    # the deprecated argument is ignored, but the classifier still works
    probabilities = clf.fit(instances).apply(instances).features
    assert np.all(probabilities > 0)
    assert np.all(probabilities < 1)


def test_parse_module_svm_resolves_to_calibrated_svc():
    module = parse_module(ConfigValue.wrap([], {'method': 'svm'}), Path('/'))
    assert isinstance(module, CalibratedSVC)
    assert isinstance(module.estimator, CalibratedClassifierCV)


def test_parse_module_svm_with_probability_still_parses():
    with pytest.warns(DeprecationWarning, match='`probability` is deprecated'):
        module = parse_module(ConfigValue.wrap([], {'method': 'svm', 'probability': True}), Path('/'))
    assert isinstance(module, CalibratedSVC)
