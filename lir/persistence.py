import pickle
from os import PathLike
from pathlib import Path

from lir.aggregation import Aggregation, AggregationData
from lir.config.base import ContextAwareDict, check_is_empty, config_parser, pop_field
from lir.lrsystems.lrsystems import LRSystem


def load_model(path: Path) -> LRSystem:
    """
    Load previously cached model.

    Parameters
    ----------
    path : Path
        The path to the .pkl file containing the model.

    Returns
    -------
    LRSystem
        The loaded model.
    """
    try:
        with open(path, 'rb') as f:
            # It is assumed exclusively `LRSystem` models will be loaded, which are considered safe
            return pickle.load(f)  # noqa: S301
    except Exception:
        raise RuntimeError('Could not load model from .pkl file')


def save_model(path: Path, model: LRSystem) -> None:
    """
    Save a model to disk.

    Parameters
    ----------
    path : Path
        The path to the .pkl file where the model should be saved.
    model : LRSystem
        The model to be saved.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(pickle.dumps(model))


class SaveModel(Aggregation):
    """
    Write the model to a file.

    Parameters
    ----------
    output_dir : Path
        The directory where the model should be written.
    filename : PathLike | str
        The filename to be created for the model.
    """

    def __init__(self, output_dir: Path, filename: PathLike | str = 'model.pkl') -> None:
        """
        Initialize the aggregation object.

        The model is saved as a pickle file, in a file named `filename`, that is written to a subdirectory of
        `output_dir`, that is created for each run.

        If `filename` is an absolute path, or if `filename` is relative to `output_dir`, then the model is saved to this
        file as-is, instead of to a file in a newly created subdirectory.

        Parameters
        ----------
        output_dir : Path
            The directory where the model should be written.
        filename : PathLike | str
            The filename to be created for the model.
        """
        self.output_dir = output_dir
        self.filename = Path(filename)

    def report(self, data: AggregationData) -> None:
        """
        Create a directory for the run and write the trained LR system model to file.

        Parameters
        ----------
        data : AggregationData
            The data to be aggregated, containing the trained LR system model and the run name.
        """
        if self.filename.is_absolute() or self.filename.is_relative_to(self.output_dir):
            save_model(self.filename, data.lrsystem)
        else:
            dirname = self.output_dir / data.run_name if data.run_name else self.output_dir
            dirname.mkdir(parents=True, exist_ok=True)
            save_model(dirname / self.filename, data.lrsystem)


@config_parser
def parse_save_model(config: ContextAwareDict, output_dir: Path) -> SaveModel:
    """
    Parse a configuration section that describes how a `SaveModel` instance should be instantiated.

    Parameters
    ----------
    config : ContextAwareDict
        The configuration section.
    output_dir : Path
        Directory where the instantiated object may write its output.

    Returns
    -------
    SaveModel
        The instantiated object.
    """
    filename = pop_field(config, 'filename', default='model.pkl', validate=str)
    check_is_empty(config)
    return SaveModel(output_dir, filename)
