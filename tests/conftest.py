import pytest

from lir.algorithms.bootstraps import BootstrapAtData
from lir.algorithms.logistic_regression import LogitCalibrator
from lir.data.datasets.synthesized_normal_binary import SynthesizedNormalBinaryData, SynthesizedNormalDataClass
from lir.data.models import LLRData
from lir.transform import as_transformer


@pytest.fixture
def synthesized_normal_data() -> SynthesizedNormalBinaryData:
    data_classes = {
        1: SynthesizedNormalDataClass(mean=0, std=1, size=100),  # H1
        0: SynthesizedNormalDataClass(mean=2, std=1, size=100),  # H2
    }

    return SynthesizedNormalBinaryData(data_classes=data_classes, seed=42)


@pytest.fixture
def synthesized_llrs(synthesized_normal_data: SynthesizedNormalBinaryData) -> LLRData:
    data = synthesized_normal_data.get_instances()
    return as_transformer(LogitCalibrator()).fit_transform(data).replace_as(LLRData)


@pytest.fixture
def synthesized_llrs_with_interval(synthesized_normal_data: SynthesizedNormalBinaryData) -> LLRData:
    bootstrap = BootstrapAtData(steps=[('clf', LogitCalibrator())])
    data = synthesized_normal_data.get_instances()
    return bootstrap.fit_transform(data)
