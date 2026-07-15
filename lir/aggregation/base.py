import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import NamedTuple

from lir.config.substitution import HyperparameterOption
from lir.data.models import LLRData
from lir.lrsystems.lrsystems import LRSystem


LOG = logging.getLogger(__name__)


def _resolve_path(basedir: Path, filename: PathLike | str | Path) -> Path:
    filename = Path(filename)
    if filename.is_absolute() or filename.is_relative_to(basedir):
        return filename

    path = basedir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class AggregationData(NamedTuple):  # numpydoc ignore=PR02
    """
    Representation of aggregated data.

    Parameters
    ----------
    llrdata : LLRData
        The LLR data containing LLRs and labels.
    lrsystem : LRSystem
        The model that produced the results.
    parameters : dict[str, Any]
        Parameters that identify the system producing the results.
    run_name : str
        String representation of the run that produced the results.
    experiment_output_dir : Path
        The directory where the results should be stored for this run.
    get_full_fit_lrsystem : Callable[[], LRSystem] | None
        Optional callable that lazily provides a model fitted on full data (ignoring splits).
    """

    llrdata: LLRData
    lrsystem: LRSystem
    parameters: dict[str, HyperparameterOption | str]
    run_name: str
    experiment_output_dir: Path
    get_full_fit_lrsystem: Callable[[], LRSystem] | None = None

    def resolve_path_for_experiment(self, filename: Path | PathLike | str) -> Path:
        """
        Obtain the full path for a filename and make sure it is a sub path of ``experiment_output_dir``.

        If the filename is an absolute path, or the filename is relative to ``experiment_output_dir``, return the
        filename as-is.

        Otherwise, construct a path for the filename relative to ``experiment_output_dir``.

        Parameters
        ----------
        filename : Path | PathLike | str
            A file or directory.

        Returns
        -------
        Path
            A path relative to the output directory for the experiment.
        """
        return _resolve_path(self.experiment_output_dir, filename)

    def resolve_path_for_run(self, filename: Path | PathLike | str) -> Path:
        """
        Obtain the full path for a filename and make sure it is a sub path of ``experiment_output_dir``.

        If the filename is an absolute path, or the filename is relative to ``experiment_output_dir``, return the
        filename as-is.

        Otherwise, construct a path for the filename relative to ``experiment_output_dir``.

        Parameters
        ----------
        filename : Path | PathLike | str
            A file or directory.

        Returns
        -------
        Path
            A path relative to the output directory for the experiment.
        """
        return _resolve_path(self.experiment_output_dir / self.run_name, filename)


class Aggregation(ABC):
    """
    Base representation of an aggregated data collection.

    Other classes may extend from this class.
    """

    @abstractmethod
    def report(self, data: AggregationData) -> None:
        """
        Report that new results are available.

        Parameters
        ----------
        data : AggregationData
            The aggregated data to be reported.
        """
        raise NotImplementedError

    def close(self) -> None:  # noqa: B027
        """
        Finalize the aggregation; no more results will come in.

        The close method is called at the end of gathering the aggregation(s) to ensure files are closed, buffers are
        cleared, or other things that need to finish / tear down.
        """
