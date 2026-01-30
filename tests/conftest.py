import pytest

from lir.algorithms.bootstraps import BootstrapAtData
from lir.algorithms.logistic_regression import LogitCalibrator
from lir.data.datasets.synthesized_normal_binary import SynthesizedNormalBinaryData, SynthesizedNormalData
from lir.data.models import FeatureData, LLRData
from lir.transform import as_transformer


@pytest.fixture
def synthesized_normal_data() -> FeatureData:
    h1_data = SynthesizedNormalData(mean=0, std=1, size=100)  # H1
    h2_data = SynthesizedNormalData(mean=2, std=1, size=100)  # H2
    return SynthesizedNormalBinaryData(h1_data, h2_data, seed=42).get_instances()


@pytest.fixture
def synthesized_llrs(synthesized_normal_data: SynthesizedNormalBinaryData) -> LLRData:
    data = synthesized_normal_data.get_instances()
    return as_transformer(LogitCalibrator()).fit_apply(data).replace_as(LLRData)


@pytest.fixture
def synthesized_llrs_with_interval(synthesized_normal_data: FeatureData) -> LLRData:
    bootstrap = BootstrapAtData(steps=[('clf', LogitCalibrator())])
    return bootstrap.fit_apply(synthesized_normal_data)
