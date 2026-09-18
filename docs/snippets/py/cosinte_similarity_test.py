import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from lir import PairedFeatureData

from .cosine_similarity import CosineSimilarity


@pytest.mark.parametrize(
    'features,expected_similarities',
    [
        (np.array([[[3, 2, 0, 5], [1, 0, 0, 0]]]), np.array([0.48666426])),
        (np.array([[1, 1, 1, 1], [1, 0, 0, 1], [-1, 1, 1, -1]]).reshape(-1, 2, 2), np.array([1, 0, -1])),
    ],
)
def test_cosine_similarity(features: np.ndarray, expected_similarities: np.ndarray):
    cosim = CosineSimilarity()
    similarities = cosim.apply(PairedFeatureData(features=features, n_trace_instances=1, n_ref_instances=1)).features[
        :, 0
    ]
    sklearn_similarities = np.diagonal(cosine_similarity(features[:, 0, :], features[:, 1, :]))
    assert np.allclose(similarities, sklearn_similarities)
    assert np.allclose(similarities, expected_similarities)
