import pytest

from lir import metrics
from lir.data.data_strategies import MulticlassCrossValidation
from lir.data.datasets.synthesized_normal_multiclass import (
    SynthesizedNormalMulticlassData,
    SynthesizedDimension,
)
from lir.lrsystems.lrsystems import LLRData
from lir.lrsystems.two_level import TwoLevelSystem
from lir.transform.pairing import SourcePairing


def _calculate_cllr(
    mean: float = 0.0, std: float = 1.0, error_std: float = 1.0
) -> float:
    params = {
        "population_size": 20,
        "sources_size": 6,
        "dimensions": [SynthesizedDimension(mean, std, error_std)],
        "seed": 0,
    }

    data = SynthesizedNormalMulticlassData(**params)
    data = MulticlassCrossValidation(data, 2)

    pairing = SourcePairing(seed=0)

    training_data, test_data = next(iter(data))

    system = TwoLevelSystem(
        "test_system", None, pairing, None, n_trace_instances=50, n_ref_instances=50
    )
    system.fit(training_data)
    llr_data: LLRData = system.apply(test_data)
    pair_llrs = llr_data.llrs
    pair_labels = llr_data.labels

    return metrics.cllr(pair_llrs, pair_labels)


def test_two_level_system():
    # identical parameter values should yield the same results
    assert _calculate_cllr(error_std=1) == _calculate_cllr(error_std=1)

    # extremely large variation should yield an non-informative system with CLLR=1
    assert _calculate_cllr(error_std=1000) == pytest.approx(1, abs=0.01)

    # increasing variation should reduce performance (incease CLLR)
    for i in [0.01, 0.1, 1]:
        assert _calculate_cllr(error_std=i) < _calculate_cllr(error_std=i * 10)

    assert _calculate_cllr(error_std=0.01) == pytest.approx(0.017634552368655944, abs=0.001)
