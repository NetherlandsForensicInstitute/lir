from pathlib import Path
from pickle import UnpicklingError
from unittest import mock

import confidence
import pytest
from _pytest.tmpdir import TempPathFactory
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from lir.config.base import _expand
from lir.data.datasets.synthesized_normal_binary import SynthesizedNormalBinaryData, SynthesizedNormalDataClass
from lir.data.models import FeatureData
from lir.lrsystems.binary_lrsystem import BinaryLRSystem
from lir.lrsystems.lrsystems import LRSystem
from lir.persistence import save_model, load_model, parse_save_model
from lir.transform import as_transformer
from lir.transform.pipeline import Pipeline
from lir.util import probability_to_logodds


@pytest.fixture
def sample_feature_data() -> FeatureData:
    """Provide a synthesized data set."""
    return SynthesizedNormalBinaryData({0: SynthesizedNormalDataClass(-1, 1, 100), 1: SynthesizedNormalDataClass(1, 1, 100)}, seed=0).get_instances()


@pytest.fixture
def trained_lr_system(sample_feature_data: FeatureData) -> LRSystem:
    """Provide a basic trained LR system model based on specific settings and data."""
    pipeline = Pipeline(steps=[
        ("scaler", as_transformer(StandardScaler())),
        ("clf", as_transformer(LogisticRegression(class_weight='balanced'))),
        ("to_llr", as_transformer(probability_to_logodds)),
    ])
    lrsystem = BinaryLRSystem(name="lrsystem", pipeline=pipeline)
    lrsystem.fit(sample_feature_data)

    return lrsystem


@pytest.fixture
def model_file_path(tmp_path_factory: TempPathFactory) -> Path:
    """Return a temporary file name for a saved model."""
    return tmp_path_factory.mktemp("model") / "model.pkl"


def test_serialize_trained_lr_system(trained_lr_system: LRSystem, model_file_path: Path):
    """Check that a trained LR system can be serialized."""
    # When we serialize the LR system
    save_model(model_file_path, trained_lr_system)

    # There should be a file we can load
    assert model_file_path.exists()


def test_deserialize_trained_lr_system(trained_lr_system: LRSystem, sample_feature_data: FeatureData, model_file_path: Path):
    """Check that a deserialized, trained LR system yields exactly the same results."""
    # Given that we have a certain LR system serialized
    save_model(model_file_path, trained_lr_system)

    # When the model is deserialized
    deserialized_model = load_model(model_file_path)

    # The deserialized model and the model it originated from should be of the same type of LR system
    assert type(trained_lr_system) is type(deserialized_model)

    # The calculated LLR output should be identical to the LR system output of the serialized model
    expected_llr_data = trained_lr_system.apply(sample_feature_data)
    deserialized_model_data = deserialized_model.apply(sample_feature_data)

    assert deserialized_model_data == expected_llr_data


def test_deserialize_from_invalid_pickle_file(trained_lr_system: LRSystem, model_file_path: Path):
    """Check that an appropriate error is raised when unable to unpickle serialized model."""
    # Given that we have a serialized model for a given type of `ModelSettings`
    save_model(model_file_path, trained_lr_system)

    with mock.patch('pickle.load', side_effect=UnpicklingError("Some pickle error")):
        # When pickle can't load the given file, we expect an appropriate error to be raised
        with pytest.raises(RuntimeError) as exception_info:
            load_model(model_file_path)

        assert "Could not load model from .pkl file" in str(exception_info.value)


@pytest.mark.parametrize("yaml", [
    "",
    "filename: yolo.pkl"
])
def test_config_parser(yaml: str):
    config = _expand([], confidence.loads(yaml))
    parse_save_model().parse(config, output_dir=Path("/"))
