from pathlib import Path

from lir import DataProvider
from lir.aggregation import Aggregation, AggregationData
from lir.algorithms.logistic_regression import LogitCalibrator
from lir.experiments.inference import InferenceExperiment
from lir.lrsystems import BinaryLRSystem


class CatchOutput(Aggregation):
    data: AggregationData

    def report(self, data: AggregationData):
        self.data = data


def test_fitted_lrsystem(synthesized_normal_data_provider: DataProvider):
    lrsystem = BinaryLRSystem(LogitCalibrator().fit(synthesized_normal_data_provider.get_instances()))
    output = CatchOutput()

    exp = InferenceExperiment(
        Path('/'), lrsystem=lrsystem, output=output, inference_data=synthesized_normal_data_provider
    )

    exp.run()
    assert len(output.data.llrdata) == len(synthesized_normal_data_provider.get_instances())
