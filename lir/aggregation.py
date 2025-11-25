import csv
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import IO, Any

from matplotlib import pyplot as plt

from lir.data.models import LLRData
from lir.plotting import Canvas, axes, ece, llr_interval, lr_histogram, pav, savefig


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


class AggregatePlot(Aggregation):
    """Aggregation that generates plots by repeatedly calling a plotting function."""

    def __init__(self, plot_function: Callable, output_dir: str) -> None:
        super().__init__()

        self.f = plot_function
        self.dir = output_dir
        self.plot_type = plot_function.__name__
        self._fig, self._ax = plt.subplots(figsize=(10, 8))
        self._canvas = Canvas(self._ax)

    def report(self, llrdata: LLRData, parameters: dict[str, Any]) -> None:
        self._canvas.plot(
            [],
            [],
            marker='None',
            linestyle='None',
            color='white',  # This is necessary to avoid matplotlib from cycling through colours
            label=', '.join(f'{k}={v}' for k, v in parameters.items()),
        )  # Dummy plot to add legend entry

        self.f(None, llrdata, self._canvas)

    def close(self) -> None:
        """Generate and save each plot after all results have been reported."""
        self._ax.set_title(f'Aggregated {self.plot_type}')
        self._fig.savefig(f'{self.dir}/aggregated_{self.plot_type}.png')


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
    output_path: Path

    def __init__(self, output_dir: str) -> None:
        self.output_path = Path(output_dir)
        self.ax = axes(self.output_path)

    def report(self, llrdata: LLRData, parameters: dict[str, Any]) -> None:
        """Helper function to generate and save a PAV-plot to the output directory."""
        dir_name = self.output_path
        file_name = dir_name / f'{self.__class__.__name__}.png'
        dir_name.mkdir(exist_ok=True, parents=True)
        with savefig(str(file_name)) as _:
            self._plot(llrdata)

    @staticmethod
    def _plot(llrs: LLRData, **kwargs: Any) -> None:
        raise NotImplementedError

    def close(self) -> None:
        plt.close()


class PlotPAV(Plot):
    @staticmethod
    def _plot(llrs: LLRData, **kwargs: Any) -> None:
        pav(llrdata=llrs, **kwargs)


class PlotECE(Plot):
    @staticmethod
    def _plot(llrs: LLRData, **kwargs: Any) -> None:
        ece(llrdata=llrs, **kwargs)


class PlotLRHistogram(Plot):
    @staticmethod
    def _plot(llrs: LLRData, **kwargs: Any) -> None:
        lr_histogram(llrdata=llrs, **kwargs)


class PlotLLRInterval(Plot):
    @staticmethod
    def _plot(llrs: LLRData, **kwargs: Any) -> None:
        llr_interval(llrdata=llrs, **kwargs)
