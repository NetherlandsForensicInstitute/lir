import numpy as np
import pytest

from lir import DataStrategy, FeatureData
from lir.datasets.synthesized_normal_binary import SynthesizedNormalBinaryData, SynthesizedNormalData
from lir.datasets.synthesized_normal_multiclass import SynthesizedDimension, SynthesizedNormalMulticlassData
from lir.splitting_strategies import (
    AutoCrossValidation,
    AutoTrainTestSplit,
    CrossValidation,
    PairsTrainTestSplit,
    PredefinedTrainTestSplit,
    SourcesCrossValidation,
    SourcesTrainTestSplit,
    TrainTestSplit,
)


@pytest.mark.parametrize(
    'instances,auto_strategy,explicit_strategy',
    [
        (
            SynthesizedNormalBinaryData(
                SynthesizedNormalData(-1, 1, 10), SynthesizedNormalData(1, 1, 10)
            ).get_instances(),
            AutoTrainTestSplit(seed=0),
            TrainTestSplit(test_size=0.5, seed=0),
        ),
        (
            SynthesizedNormalBinaryData(
                SynthesizedNormalData(-1, 1, 10), SynthesizedNormalData(1, 1, 10)
            ).get_instances(),
            AutoCrossValidation(folds=5),
            CrossValidation(folds=5),
        ),
        (
            SynthesizedNormalMulticlassData(
                num_source_ids=10,
                num_sources_per_source_id=3,
                seed=0,
                dimensions=[SynthesizedDimension(0, 1, 0.2), SynthesizedDimension(0, 1, 0.2)],
            ).get_instances(),
            AutoTrainTestSplit(seed=0),
            SourcesTrainTestSplit(test_size=0.5, seed=0),
        ),
        (
            SynthesizedNormalMulticlassData(
                num_source_ids=10,
                num_sources_per_source_id=3,
                seed=0,
                dimensions=[SynthesizedDimension(0, 1, 0.2), SynthesizedDimension(0, 1, 0.2)],
            ).get_instances(),
            AutoCrossValidation(folds=5),
            SourcesCrossValidation(folds=5),
        ),
        (
            FeatureData(features=np.arange(16).reshape(-1, 2), role_assignments=np.array(['train', 'test']).repeat(4)),
            AutoTrainTestSplit(),
            PredefinedTrainTestSplit(),
        ),
    ],
)
def test_reproducibility(instances: FeatureData, auto_strategy: DataStrategy, explicit_strategy: DataStrategy):
    # split the data using an automatic strategy
    splits1 = list(auto_strategy.apply(instances))

    # split the data by explicitly using the appropriate strategy
    splits2 = list(explicit_strategy.apply(instances))

    # the splits from both strategies should be equal
    for i in range(len(splits1)):
        assert np.all(splits1[i][0].features == splits2[i][0].features)


def test_auto_train_test_split_uses_pairs_strategy_for_two_column_source_ids():
    instances = FeatureData(
        features=np.arange(16).reshape(8, 2),
        source_ids=np.array([[0, 1], [0, 2], [1, 2], [3, 4], [3, 5], [4, 5], [0, 3], [1, 4]]),
    )

    auto_splits = list(AutoTrainTestSplit(test_size=0.5, seed=0).apply(instances))
    explicit_splits = list(PairsTrainTestSplit(test_size=0.5, seed=0).apply(instances))

    assert len(auto_splits) == len(explicit_splits) == 1
    assert np.array_equal(auto_splits[0][0].features, explicit_splits[0][0].features)
    assert np.array_equal(auto_splits[0][1].features, explicit_splits[0][1].features)
