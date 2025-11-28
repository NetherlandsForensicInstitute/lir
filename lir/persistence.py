import pickle
from pathlib import Path

from lir.aggregation import Aggregation, AggregationData
from lir.config.base import ContextAwareDict, check_is_empty, config_parser, pop_field
from lir.lrsystems.lrsystems import LRSystem


def load_model(path: Path) -> LRSystem:
    """Load previously cached model."""
    try:
        with open(path, 'rb') as f:
            # It is assumed exclusively `LRSystem` models will be loaded, which are considered safe
            return pickle.load(f)  # noqa: S301
    except Exception:
        raise RuntimeError('Could not load model from .pkl file')


def save_model(path: Path, model: LRSystem) -> None:
    """Save a model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(pickle.dumps(model))


class SaveModel(Aggregation):
    """
    Write the model to a file.
    """

    def __init__(self, path: Path):
        self.path = path

    def report(self, data: AggregationData) -> None:
        save_model(self.path, data.lrsystem)


@config_parser
def parse_save_model(config: ContextAwareDict, output_dir: Path) -> SaveModel:
    """
    Parse a configuration section that describes how a `SaveModel` instance should be instantiated.

    :param config: the configuration section
    :param output_dir: directory where the instantiated object may write its output
    """
    filename = pop_field(config, 'filename', default='model.pkl', validate=str)
    check_is_empty(config)
    return SaveModel(output_dir / filename)
