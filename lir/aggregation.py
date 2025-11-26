import csv
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import IO, Any

import numpy as np
from matplotlib import pyplot as plt

from lir.algorithms.bayeserror import plot_nbe
from lir.data.models import LLRData
from lir.plotting import calibrator_fit, llr_interval, lr_histogram, pav, score_distribution, tippett
from lir.plotting.expected_calibration_error import plot_ece


class Aggregation(ABC):
    @abstractmethod
    def report(self, llrdata: LLRData, parameters: dict[str, Any]) -> None:
        """
        Report that new results are available.

        :param llrdata: the LLR data containing LLRs and labels.
        :param parameters: parameters that identify the system producing the results
        """
        raise NotImplementedError

    def close(self) -> None:  # noqa: B027
        """
        Finalize the aggregation; no more results will come in.

        The close method is called at the end of gathering the aggregation(s) to ensure files are closed, buffers are
        cleared, or other things that need to finish / tear down.
        """
        pass


class WriteMetricsToCsv(Aggregation):
    def __init__(self, output_dir: Path, metrics: Mapping[str, Callable]):
        self.path = output_dir / 'metrics.csv'
        self._file: IO[Any] | None = None
        self._writer: csv.DictWriter | None = None
        self.metrics = metrics

    def report(self, llrdata: LLRData, parameters: dict[str, Any]) -> None:
        metrics = [(key, metric(llrdata.llrs, llrdata.labels)) for key, metric in self.metrics.items()]
        results = OrderedDict(list(parameters.items()) + metrics)

        # Record column header names only once to the CSV
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, 'w')  # noqa: SIM115
            self._writer = csv.DictWriter(self._file, fieldnames=results.keys())
            self._writer.writeheader()
        self._writer.writerow(results)
        self._file.flush()  # type: ignore

    def close(self) -> None:
        if self._file:
            self._file.close()


class Plot(Aggregation):
    output_path: Path | None = None

    def __init__(self, output_dir: str | None = None) -> None:
        if output_dir:
            self.output_path = Path(output_dir)
        ax, fig = plt.gca(), plt.gcf()
        self.ax = ax
        self.fig = fig

    def report(self, llrdata: LLRData, parameters: dict[str, Any]) -> None:
        """Plot the data when new results are available.

        _plot is not called directly, to allow for easier aggregation in the future."""
        self._plot(llrdata)

    @staticmethod
    def _plot(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def close(self) -> None:
        if self.output_path is not None:
            dir_name = self.output_path
            file_name = dir_name / f'{self.__class__.__name__}.png'
            dir_name.mkdir(exist_ok=True, parents=True)

            self.fig.savefig(file_name)
        self.fig.clf()

    def plot(self, *args: Any, **kwargs: Any) -> Any:
        """Allows for direct plotting, outside of the aggregation framework."""
        self._plot(*args, **kwargs)
        self.close()


class PlotPAV(Plot):
    @staticmethod
    def _plot(llrs: LLRData, **kwargs: Any) -> None:
        pav(llrs, **kwargs)


class PlotECE(Plot):
    @staticmethod
    def _plot(llrs: LLRData, **kwargs: Any) -> None:
        plot_ece(llrs, **kwargs)


class PlotLRHistogram(Plot):
    @staticmethod
    def _plot(llrs: LLRData, **kwargs: Any) -> None:
        lr_histogram(llrs, **kwargs)


class PlotLLRInterval(Plot):
    @staticmethod
    def _plot(llrs: LLRData, **kwargs: Any) -> None:
        llr_interval(llrs, **kwargs)


class PlotNBE(Plot):
    @staticmethod
    def _plot(llrs: LLRData, **kwargs: Any) -> None:
        plot_nbe(llrs, **kwargs)


class PlotTippett(Plot):
    @staticmethod
    def _plot(llrs: LLRData, **kwargs: Any) -> None:
        tippett(llrs, **kwargs)


class PlotCalibratorFit(Plot):
    @staticmethod
    def _plot(calibrator: Any, **kwargs: Any) -> None:
        calibrator_fit(calibrator, **kwargs)


class PlotScoreDistribution(Plot):
    @staticmethod
    def _plot(scores: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        score_distribution(scores, y, **kwargs)
