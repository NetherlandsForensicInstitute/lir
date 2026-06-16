import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from lir.config.substitution import HyperparameterOption
from lir.data.models import LLRData
from lir.lrsystems.lrsystems import LRSystem


LOG = logging.getLogger(__name__)


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
    get_full_fit_lrsystem : Callable[[], LRSystem] | None
        Optional callable that lazily provides a model fitted on full data (ignoring splits).
    """

    llrdata: LLRData
    lrsystem: LRSystem
    parameters: dict[str, HyperparameterOption | str]
    run_name: str
    get_full_fit_lrsystem: Callable[[], LRSystem] | None = None


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

    @staticmethod
    def _resolve_output_path(output_dir: Path, filename: Path, run_name: str) -> Path:
        if filename.is_absolute() or filename.is_relative_to(output_dir):
            return filename

        dirname = output_dir / run_name if run_name else output_dir
        dirname.mkdir(parents=True, exist_ok=True)
        return dirname / filename
