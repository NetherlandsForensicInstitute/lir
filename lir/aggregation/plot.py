import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt

from lir.aggregation import Aggregation, AggregationData


LOG = logging.getLogger(__name__)


class PlotEach(Aggregation):
    """
    Aggregation that generates a plot for each call to ``report()``.

    Repeated calls to ``report()`` will result in separate plots.

    Parameters
    ----------
    plot_fn : Callable
        The plotting function to be used for generating plots.
    plot_name : str
        The name of the plot.
    output_path : Path | None, optional
        The directory where the plots will be saved. If `None`, plots are not saved.
    **kwargs : Any
        Additional arguments to be passed to the plotting function.

    Attributes
    ----------
    output_path : Path | None
        The directory where the plots will be saved. If `None`, plots are not saved.
    plot_fn : Callable
        The plotting function to be used for generating plots.
    plot_name : str
        The name of the plot.
    plot_fn_args : dict[str, Any]
        Additional arguments to be passed to the plotting function.
    """

    def __init__(self, plot_fn: Callable, plot_name: str, output_path: Path | None = None, **kwargs: Any) -> None:
        self.output_path = output_path
        self.plot_fn = plot_fn
        self.plot_name = plot_name
        self.plot_fn_args = kwargs

    def report(self, data: AggregationData) -> None:
        """
        Plot the data when new results are available.

        Parameters
        ----------
        data : AggregationData
            The aggregated data to be plotted.
        """
        fig, ax = plt.subplots()

        llrdata = data.llrdata
        run_name = data.run_name

        try:
            self.plot_fn(llrdata=llrdata, ax=ax, **self.plot_fn_args)
        except ValueError as e:
            LOG.warning(f'Could not generate plot {self.plot_name} for run `{run_name}`: {e}')
            return

        # Only save the figure when an output path is provided.
        if self.output_path is not None:
            dir_name = self.output_path
            plot_arguments = '_'.join(f'{k}={v}' for k, v in self.plot_fn_args.items()) if self.plot_fn_args else ''

            file_name = dir_name / run_name / f'{self.plot_name}{plot_arguments}.png'
            file_name.parent.mkdir(exist_ok=True, parents=True)

            LOG.info(f'Saving plot {self.plot_name} for run `{run_name}` to {file_name}')
            fig.savefig(file_name)

        plt.close(fig)
