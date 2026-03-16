import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt

from lir import plotting
from lir.aggregation import Aggregation, AggregationData
from lir.algorithms.bayeserror import plot_nbe
from lir.algorithms.invariance_bounds import plot_invariance_delta_functions
from lir.algorithms.llr_overestimation import plot_llr_overestimation
from lir.config.base import ContextAwareDict, config_parser, pop_field


LOG = logging.getLogger(__name__)


class AggregatePlot(Aggregation):
    """
    Aggregation that generates plots by repeatedly calling a plotting function.

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


@config_parser(reference=plotting.pav)
def pav(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    """
    Corresponding registry function to generate aggregate PAV plot.

    Parameters
    ----------
    config : ContextAwareDict
        The configuration dictionary for the plot.
    output_dir : Path
        The directory where the plot will be saved.

    Returns
    -------
    AggregatePlot
        An instance of the AggregatePlot class configured to generate PAV plots.
    """
    plot_name = pop_field(config, 'plot_name', default='PAV')
    return AggregatePlot(plotting.pav, plot_name, output_dir, **config)


@config_parser(reference=plotting.ece)
def ece(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    """
    Corresponding registry function to generate aggregate ECE plot.

    Parameters
    ----------
    config : ContextAwareDict
        The configuration dictionary for the plot.
    output_dir : Path
        The directory where the plot will be saved.

    Returns
    -------
    AggregatePlot
        An instance of the AggregatePlot class configured to generate ECE plots.
    """
    plot_name = pop_field(config, 'plot_name', default='ECE')
    return AggregatePlot(plotting.ece, plot_name, output_dir, **config)


@config_parser(reference=plotting.lr_histogram)
def lr_histogram(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    """
    Corresponding registry function to generate aggregate LR Histogram.

    Parameters
    ----------
    config : ContextAwareDict
        The configuration dictionary for the plot.
    output_dir : Path
        The directory where the plot will be saved.

    Returns
    -------
    AggregatePlot
        An instance of the AggregatePlot class configured to generate LR Histogram plots.
    """
    plot_name = pop_field(config, 'plot_name', default='LR_Histogram')
    return AggregatePlot(plotting.lr_histogram, plot_name, output_dir, **config)


@config_parser(reference=plotting.llr_interval)
def llr_interval(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    """
    Corresponding registry function to generate aggregate LLR interval plot.

    Parameters
    ----------
    config : ContextAwareDict
        The configuration dictionary for the plot.
    output_dir : Path
        The directory where the plot will be saved.

    Returns
    -------
    AggregatePlot
        An instance of the AggregatePlot class configured to generate LLR interval plots.
    """
    plot_name = pop_field(config, 'plot_name', default='LLR_Interval')
    return AggregatePlot(plotting.llr_interval, plot_name, output_dir, **config)


@config_parser(reference=plot_llr_overestimation)
def llr_overestimation(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    """
    Corresponding registry function to generate aggregate LLR overestimation plot.

    Parameters
    ----------
    config : ContextAwareDict
        The configuration dictionary for the plot.
    output_dir : Path
        The directory where the plot will be saved.

    Returns
    -------
    AggregatePlot
        An instance of the AggregatePlot class configured to generate LLR overestimation plots.
    """
    plot_name = pop_field(config, 'plot_name', default='LLR_Overestimation')
    return AggregatePlot(plot_llr_overestimation, plot_name, output_dir, **config)


@config_parser(reference=plot_nbe)
def nbe(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    """
    Corresponding registry function to generate aggregate NBE plot.

    Parameters
    ----------
    config : ContextAwareDict
        The configuration dictionary for the plot.
    output_dir : Path
        The directory where the plot will be saved.

    Returns
    -------
    AggregatePlot
        An instance of the AggregatePlot class configured to generate NBE plots.
    """
    plot_name = pop_field(config, 'plot_name', default='NBE')
    return AggregatePlot(plot_nbe, plot_name, output_dir, **config)


@config_parser(reference=plotting.tippett)
def tippett(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    """
    Corresponding registry function to generate aggregate Tippett plot.

    Parameters
    ----------
    config : ContextAwareDict
        The configuration dictionary for the plot.
    output_dir : Path
        The directory where the plot will be saved.

    Returns
    -------
    AggregatePlot
        An instance of the AggregatePlot class configured to generate Tippett plots.
    """
    plot_name = pop_field(config, 'plot_name', default='Tippett')
    return AggregatePlot(plotting.tippett, plot_name, output_dir, **config)


@config_parser(reference=plot_invariance_delta_functions)
def invariance_delta_function(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    """
    Corresponding registry function to generate aggregate invariance delta function plot.

    Parameters
    ----------
    config : ContextAwareDict
        The configuration dictionary for the plot.
    output_dir : Path
        The directory where the plot will be saved.

    Returns
    -------
    AggregatePlot
        An instance of the AggregatePlot class configured to generate invariance delta function plots.
    """
    plot_name = pop_field(config, 'plot_name', default='Invariance_Delta_Functions')
    return AggregatePlot(plot_invariance_delta_functions, plot_name, output_dir, **config)
