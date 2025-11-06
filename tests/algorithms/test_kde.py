import numpy as np

from lir.algorithms.kde import KDECalibrator


def test_kde_dimensions():
    features = np.random.normal(loc=0, scale=1, size=(10, 1))
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    kde = KDECalibrator(bandwidth=1)
    kde.fit(features, labels).transform(features)
    kde.fit(features.flatten(), labels).transform(features.flatten())
