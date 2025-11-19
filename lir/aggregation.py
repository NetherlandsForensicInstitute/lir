import csv
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import IO, Any

from matplotlib import pyplot as plt

from lir.data.models import LLRData
from lir.plotting import Canvas


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

        self.plots: dict[str, Any] = {}  # Dictionary to hold figures and axes for different plot types
        # Add initial plot setup for this plot_type
        self._setup_plot()

    def _setup_plot(self) -> None:
        """Set up a new plot for the given plot type."""
        fig, ax = plt.subplots()
        canvas = Canvas(ax)
        self.plots = {'fig': fig, 'ax': ax, 'canvas': canvas, 'legend_suffix': []}

    def report(self, llrdata: LLRData, parameters: dict[str, Any]) -> None:
        self.plots['canvas'].plot(
            [],
            [],
            marker='None',
            linestyle='None',
            label=', '.join(f'{k}={v}' for k, v in parameters.items()),
        )  # Dummy plot to add legend entry

        self.f(None, llrdata, self.plots['canvas'])

    def close(self) -> None:
        """Generate and save each plot after all results have been reported."""
        ax = self.plots['ax']
        ax.set_title(f'Aggregated {self.plot_type}')
        self.plots['fig'].savefig(f'{self.dir}/aggregated_{self.plot_type}.png')


class WriteMetricsToCsv(Aggregation):
    def __init__(self, path: Path, metrics: Mapping[str, Callable]):
        self.path = path
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
