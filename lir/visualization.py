from collections.abc import Callable
from pathlib import Path
from typing import Any

from lir.data.models import LLRData
from lir.plotting import savefig


def _save_or_plot(
    ax: Any | None, base_path: Path | None, filename: str, plot_func: Callable[[Any, Any], None], *args: Any
) -> None:
    """
    Plot to an axis or save plot to file.

    Parameters
    ----------
    ax : Any | None
        The axis to plot on, if provided.
    base_path : Path | None
        The base path to save the plot, if ax is not provided.
    filename : str
        The filename to save the plot as, if ax is not provided.
    plot_func : Callable[[Any, Any], None]
        The function that generates the plot, taking an axis or figure and data as arguments.
    *args : Any
        Additional arguments to pass to the plot function.
    """
    if ax is not None:
        plot_func(ax, *args)
    elif base_path is not None:
        base_path.mkdir(exist_ok=True, parents=True)
        path = base_path / filename
        with savefig(str(path)) as fig:
            plot_func(fig, *args)
    else:
        raise ValueError('Either base_path or ax must be provided.')


def pav(base_path: Path | None, llrdata: LLRData, ax: Any | None = None) -> None:
    """
    Generate and handle a PAV plot, either saving to a file or plotting on a given axis.

    Parameters
    ----------
    base_path : Path | None
        The base path to save the plot, if ax is not provided.
    llrdata : LLRData
        The data to be plotted.
    ax : Any | None, optional
        The axis to plot on, if provided. If None, the plot will be saved to a file, by default None.
    """
    _save_or_plot(ax, base_path, 'pav.png', lambda obj, data: obj.pav(data), llrdata)


def ece(base_path: Path | None, llrdata: LLRData, ax: Any | None = None) -> None:
    """
    Generate and handle an ECE plot, either saving to a file or plotting on a given axis.

    Parameters
    ----------
    base_path : Path | None
        The base path to save the plot, if ax is not provided.
    llrdata : LLRData
        The data to be plotted.
    ax : Any | None, optional
        The axis to plot on, if provided. If None, the plot will be saved to a file, by default None.
    """
    _save_or_plot(ax, base_path, 'ece.png', lambda obj, data: obj.ece(data), llrdata)


def lr_histogram(base_path: Path | None, llrdata: LLRData, ax: Any | None = None) -> None:
    """
    Generate and handle a histogram plot of likelihood ratios, either saving to file or plotting on an axis.

    Parameters
    ----------
    base_path : Path | None
        The base path to save the plot, if ax is not provided.
    llrdata : LLRData
        The data to be plotted.
    ax : Any | None, optional
        The axis to plot on, if provided. If None, the plot will be saved to a file, by default None.
    """
    _save_or_plot(ax, base_path, 'histogram.png', lambda obj, data: obj.lr_histogram(data), llrdata)


def llr_interval(base_path: Path | None, llrdata: LLRData, ax: Any | None = None) -> None:
    """
    Generate and handle a Score-LR plot, either saving to file or plotting on an axis.

    Parameters
    ----------
    base_path : Path | None
        The base path to save the plot, if ax is not provided.
    llrdata : LLRData
        The data to be plotted.
    ax : Any | None, optional
        The axis to plot on, if provided. If None, the plot will be saved to a file, by default None.
    """
    _save_or_plot(ax, base_path, 'llr_interval.png', lambda obj, data: obj.llr_interval(data), llrdata)
