import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt

from lir.aggregation import Aggregation, AggregationData
from lir.config.base import ConfigParser, ContextAwareDict, pop_field
from lir.registry import _get_attribute_by_name


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
    **kwargs : Any
        Additional arguments to be passed to the plotting function.

    Attributes
    ----------
    plot_fn : Callable
        The plotting function to be used for generating plots.
    plot_name : str
        The name of the plot.
    plot_fn_args : dict[str, Any]
        Additional arguments to be passed to the plotting function.
    """

    def __init__(self, plot_fn: Callable, plot_name: str, **kwargs: Any) -> None:
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

        plot_arguments = '_'.join(f'{k}={v}' for k, v in self.plot_fn_args.items()) if self.plot_fn_args else ''
        file_name = data.resolve_path_for_run(f'{self.plot_name}{plot_arguments}.png')

        LOG.info(f'Saving plot {self.plot_name} for run `{run_name}` to {file_name}')
        fig.savefig(file_name)

        plt.close(fig)


class PlotEachConfigParser(ConfigParser):
    """
    Configuration parser for aggregate plots.

    Parameters
    ----------
    method : str
        The Python name of the plot function.
    default_plot_name : str | None
        The plot name. If `None`, the value of `method` is used.
    """

    def __init__(self, method: str, default_plot_name: str | None = None):
        self.ref_name = method
        self.plot_fn = _get_attribute_by_name(method)
        self.default_plot_name = default_plot_name or method

    def parse(self, config: ContextAwareDict, output_dir: Path) -> PlotEach:
        """
        Parse a configuration section for an aggregate plot.

        Parameters
        ----------
        config : ContextAwareDict
            Configuration section.
        output_dir : Path
            Output directory.

        Returns
        -------
        PlotEach
            Parsed aggregate plot.
        """
        plot_name = pop_field(config, 'plot_name', default=self.default_plot_name)
        return PlotEach(self.plot_fn, plot_name, **config)

    def reference(self) -> str:
        """
        Return the `method` argument of the constructor as the reference object.

        Returns
        -------
        str
            The `method` argument of the constructor.
        """
        return self.ref_name
