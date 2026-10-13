from lir import DataProvider
from lir.data_setup.inference import InferenceDataSetup


def test_fitted_lrsystem(synthesized_normal_data_provider: DataProvider):
    data = InferenceDataSetup(synthesized_normal_data_provider, synthesized_normal_data_provider)
    assert len(list(data.get_train_inference_pairs())) == 1
    assert len(list(data.get_train_inference_pairs())) == 1
    pair = next(iter(data.get_train_inference_pairs()))
    assert pair[0] is not None
    assert pair[1] is not None
    assert len(pair[0]) == len(synthesized_normal_data_provider.get_instances())
    assert len(pair[1]) == len(synthesized_normal_data_provider.get_instances())

    data = InferenceDataSetup(None, synthesized_normal_data_provider)
    pair = next(iter(data.get_train_inference_pairs()))
    assert pair[0] is None
    assert pair[1] is not None

    data = InferenceDataSetup(synthesized_normal_data_provider, None)
    pair = next(iter(data.get_train_inference_pairs()))
    assert pair[0] is not None
    assert pair[1] is None
