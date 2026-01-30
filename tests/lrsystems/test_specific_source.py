from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from lir.data.data_strategies import BinaryTrainTestSplit
from lir.data.models import FeatureData
from lir.lrsystems.binary_lrsystem import BinaryLRSystem
from lir.lrsystems.lrsystems import LLRData
from lir.transform.pipeline import Pipeline


def test_specific_source_pipeline(synthesized_normal_data: FeatureData):
    """Check that a simple Specific Source LR system can be utilized through a SKLearn pipeline."""
    splitter = BinaryTrainTestSplit(0.2, seed=0)

    steps = [
        ('preprocessing', StandardScaler()),
    ]

    pipeline = Pipeline(steps)

    specific_source_system = BinaryLRSystem(pipeline)
    data_train, data_test = next(iter(splitter.apply(synthesized_normal_data)))
    specific_source_system.fit(data_train)
    llr_data: LLRData = specific_source_system.apply(data_test)

    scores = llr_data.features
    labels = llr_data.labels

    golden_master_path = Path('tests/golden_master/test_specific_source_pipeline')
    if not Path(f'{golden_master_path}.npz').exists():
        np.savez(golden_master_path, scores=scores, labels=labels)
        pytest.skip(f'Written {golden_master_path}, skipped test for this run.')
    else:
        expected = np.load(f'{golden_master_path}.npz')
        assert np.all(scores == expected['scores'])
        assert np.all(labels == expected['labels'])
