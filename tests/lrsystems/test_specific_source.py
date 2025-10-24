import os.path

import numpy as np
import pytest

from sklearn.preprocessing import StandardScaler

from lir.data.data_strategies import BinaryTrainTestSplit
from lir.data.datasets.synthesized_normal_binary import (
    SynthesizedNormalDataClass,
    SynthesizedNormalBinaryData,
)
from lir.lrsystems.lrsystems import Pipeline, LLRData
from lir.lrsystems.specific_source import SpecificSourceSystem


@pytest.fixture
def synthesized_normal_data() -> SynthesizedNormalBinaryData:
    data_classes = {
        1: SynthesizedNormalDataClass(mean=0, std=1, size=100),  # H1
        0: SynthesizedNormalDataClass(mean=2, std=1, size=100),  # H2
    }

    return SynthesizedNormalBinaryData(data_classes=data_classes, seed=42)


def test_specific_source_pipeline(synthesized_normal_data: SynthesizedNormalBinaryData):
    """Check that a simple Specific Source LR system can be utilized through a SKLearn pipeline."""
    data = BinaryTrainTestSplit(synthesized_normal_data, 0.2, seed=0)

    steps = [
        ("preprocessing", StandardScaler()),
    ]

    pipeline = Pipeline(steps)

    specific_source_system = SpecificSourceSystem("test_system", pipeline)
    (
        (features_train, labels_train, meta_train),
        (features_test, labels_test, meta_test),
    ) = next(iter(data))
    specific_source_system.fit(features_train, labels_train, meta_train)

    llr_data: LLRData = specific_source_system.apply(
        features_test, labels_test, meta_test
    )

    scores = llr_data.llrs
    labels = llr_data.labels

    golden_master_path = "tests/golden_master/test_specific_source_pipeline"
    if not os.path.exists(f"{golden_master_path}.npz"):
        np.savez(golden_master_path, scores=scores, labels=labels)
        pytest.skip(f"Written {golden_master_path}, skipped test for this run.")
    else:
        expected = np.load(f"{golden_master_path}.npz")
        assert np.array_equal(scores, expected["scores"])
        assert np.array_equal(labels, expected["labels"])
