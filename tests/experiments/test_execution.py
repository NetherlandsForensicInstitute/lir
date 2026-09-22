from pathlib import Path

import numpy as np

from lir.config.data import DataSetup
from lir.data_strategies.labels import TrainTestSplit
from lir.datasets.synthesized_normal_binary import SynthesizedNormalBinaryData, SynthesizedNormalData
from lir.experiments.execution import DataConfig
from lir.transform.select_instances import SelectInstances


def test_filter_is_applied_before_splitting():
    provider = SynthesizedNormalBinaryData(
        SynthesizedNormalData(mean=0, std=1, size=10),
        SynthesizedNormalData(mean=2, std=1, size=10),
        seed=42,
    )
    expected = provider.get_instances().features[::2, 0]  # feature values of every other instance

    config = DataConfig(spec={}, params={}, experiment_output_dir=Path())
    config._data_setup = DataSetup(
        provider, TrainTestSplit(test_size=0.5, seed=1), SelectInstances(lambda i: i % 2 == 0)
    )

    train, test = next(iter(config.splits))
    actual = train + test

    assert len(actual) == 10
    np.testing.assert_array_equal(np.sort(actual.features.reshape(-1)), np.sort(expected))
