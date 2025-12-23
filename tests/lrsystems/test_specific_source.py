import os.path

import numpy as np
import pytest

from sklearn.preprocessing import StandardScaler

from lir.data.data_strategies import BinaryTrainTestSplit
from lir.data.datasets.synthesized_normal_binary import (
    SynthesizedNormalBinaryData,
)
from lir.lrsystems.lrsystems import LLRData
from lir.lrsystems.binary_lrsystem import BinaryLRSystem
from lir.transform.pipeline import Pipeline


def test_specific_source_pipeline(synthesized_normal_data: SynthesizedNormalBinaryData):
    """Check that a simple Specific Source LR system can be utilized through a SKLearn pipeline."""
    splitter = BinaryTrainTestSplit(0.2, seed=0)

    steps = [
        ("preprocessing", StandardScaler()),
    ]

    pipeline = Pipeline(steps)

    specific_source_system = BinaryLRSystem("test_system", pipeline)
    data_train, data_test = next(iter(splitter.apply(synthesized_normal_data.get_instances())))
    specific_source_system.fit(data_train)
    llr_data: LLRData = specific_source_system.apply(data_test)

    scores = llr_data.features
    labels = llr_data.labels

    golden_master_path = "tests/golden_master/test_specific_source_pipeline"
    if not os.path.exists(f"{golden_master_path}.npz"):
        np.savez(golden_master_path, scores=scores, labels=labels)
        pytest.skip(f"Written {golden_master_path}, skipped test for this run.")
    else:
        expected = np.load(f"{golden_master_path}.npz")
        assert np.all(scores == expected["scores"])
        assert np.all(labels == expected["labels"])
