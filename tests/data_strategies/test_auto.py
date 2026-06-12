import numpy as np
import pytest

from lir import DataStrategy, FeatureData
from lir.data_strategies import (
    AutoCrossValidation,
    AutoTrainTestSplit,
    CrossValidation,
    PredefinedTrainTestSplit,
    SourcesCrossValidation,
    SourcesTrainTestSplit,
    TrainTestSplit,
)
from lir.datasets.synthesized_normal_binary import SynthesizedNormalBinaryData, SynthesizedNormalData
from lir.datasets.synthesized_normal_multiclass import SynthesizedDimension, SynthesizedNormalMulticlassData


@pytest.mark.parametrize(
    'instances,auto_strategy,explicit_strategy',
    [
        (
            SynthesizedNormalBinaryData(
                SynthesizedNormalData(-1, 1, 10), SynthesizedNormalData(1, 1, 10)
            ).get_instances(),
            AutoTrainTestSplit(random_state=0),
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
                population_size=10,
                sources_size=3,
                seed=0,
                dimensions=[SynthesizedDimension(0, 1, 0.2), SynthesizedDimension(0, 1, 0.2)],
            ).get_instances(),
            AutoTrainTestSplit(random_state=0),
            SourcesTrainTestSplit(test_size=0.5, seed=0),
        ),
        (
            SynthesizedNormalMulticlassData(
                population_size=10,
                sources_size=3,
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
