import numpy as np

from lir.data.datasets.synthesized_normal_binary import (
    SynthesizedNormalDataClass,
    SynthesizedNormalBinaryData,
)
from lir.data.datasets.synthesized_normal_multiclass import (
    SynthesizedNormalMulticlassData,
    SynthesizedDimension,
)


def test_binary_data():
    data_spec = {
        0: SynthesizedNormalDataClass(-1, 1, 100),
        1: SynthesizedNormalDataClass(1, 1, 100),
    }
    data = SynthesizedNormalBinaryData(data_spec, seed=0)
    assert data.get_instances() == SynthesizedNormalBinaryData(data_spec, seed=0).get_instances()


def test_multiclass_data():
    dimensions = [
        SynthesizedDimension(population_mean=0, population_std=5, sources_std=1)
    ]
    data0 = SynthesizedNormalMulticlassData(
        dimensions=dimensions, population_size=100, sources_size=2, seed=0
    )
    data1 = SynthesizedNormalMulticlassData(
        dimensions=dimensions, population_size=100, sources_size=2, seed=0
    )
    assert data0.get_instances() == data1.get_instances()
    assert data0.get_instances() == data0.get_instances()
