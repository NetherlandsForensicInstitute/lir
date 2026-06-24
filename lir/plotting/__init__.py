import logging
from collections.abc import Iterator
from contextlib import _GeneratorContextManager, contextmanager
from functools import partial
from math import ceil, floor
from os import PathLike
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from lir import util
from lir.algorithms.bayeserror import plot_nbe as nbe
from lir.config.base import check_not_none
from lir.data.models import LLRData

from ..algorithms.isotonic_regression import IsotonicCalibrator
from .expected_calibration_error import plot_ece as ece


LOG = logging.getLogger(__name__)

H1_COLOR = 'red'
H2_COLOR = 'blue'


class Canvas:
    """
    Representation of an empty canvas, to be used in plotting multiple visualizations.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes instance used by wrapped plotting methods.

    Attributes
    ----------
    ax : Axes
        Matplotlib axes instance used by wrapped plotting methods.
    ece : Callable[..., Any]
        Method to plot expected calibration error (ECE) on this canvas.
    lr_histogram : Callable[..., Any]
        Method to plot a histogram of likelihood ratios on this canvas.
    nbe : Callable[..., Any]
        Method to plot the Bayes error rate (NBE) on this canvas.
    pav : Callable[..., Any]
        Method to plot the Pool Adjacent Violators (PAV) transformation on this canvas.
    score_distribution : Callable[..., Any]
        Method to plot the distribution of scores on this canvas.
    tippett : Callable[..., Any]
        Method to plot Tippett plots on this canvas.
    llr_interval : Callable[..., Any]
        Method to plot LLR intervals on this canvas.
    """

    def __init__(self, ax: Axes):
        self.ax = ax

        self.ece = partial(ece, ax)
        self.lr_histogram = partial(lr_histogram, ax)
        self.nbe = partial(nbe, ax)
        self.pav = partial(pav, ax)
        self.score_distribution = partial(score_distribution, ax)
        self.score_to_llr = partial(score_to_llr, ax)
        self.tippett = partial(tippett, ax)
        self.llr_interval = partial(llr_interval, ax)

    def title(self, label: str, **kwargs: Any) -> Any:
        """
        Set the title of the axes (wrapper for `set_title`).

        Parameters
        ----------
        label : str
            Title text.
        **kwargs : Any
            Additional keyword arguments forwarded to `Axes.set_title`.

        Returns
        -------
        Any
            Return value from `Axes.set_title`.
        """
        return self.ax.set_title(label, **kwargs)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.ax, attr)


def savefig(path: str) -> _GeneratorContextManager[Canvas]:
    """
    Create a plotting context and write the figure to a file when the context exits.

    Parameters
    ----------
    path : str
        Path to the output file. The figure is written as a PNG image.

    Returns
    -------
    _GeneratorContextManager[Canvas]
        Context manager yielding a `Canvas` and saving on exit.

    Examples
    --------
    .. code-block:: python

        with savefig(path) as ax:
            ax.pav(llrdata)

    A call to :func:`savefig` is equivalent to calling :func:`axes` with
    ``savefig=path``.
    """
    return axes(savefig=path)


def show() -> _GeneratorContextManager[Canvas]:
    """
    Create a plotting context and show the figure when the context exits.

    Returns
    -------
    _GeneratorContextManager[Canvas]
        Context manager yielding a `Canvas` and showing the figure on exit.

    Examples
    --------
    .. code-block:: python

        with show() as ax:
            ax.pav(llrdata)

    A call to :func:`show` is equivalent to calling :func:`axes` with
    ``show=True``.
    """
    return axes(show=True)


@contextmanager
def axes(savefig: PathLike | str | None = None, show: bool | None = None) -> Iterator[Canvas]:
    """
    Create a plotting context.

    Parameters
    ----------
    savefig : PathLike | str | None, optional
        File path to save the figure on context exit.
    show : bool | None, optional
        Whether to display the figure on context exit.

    Returns
    -------
    Iterator[Canvas]
        Iterator yielding a `Canvas` instance for plotting.

    Examples
    --------
    .. code-block:: python

        with axes() as ax:
            ax.pav(llrdata)
    """
    fig, ax = plt.subplots()
    try:
        yield Canvas(ax=ax)
    finally:
        if savefig:
            fig.savefig(savefig)
        if show:
            plt.show()
        plt.close(fig)


def pav(
    ax: Axes,
    llrdata: LLRData,
    add_misleading: int = 0,
    show_scatter: bool = True,
) -> None:
    """
    Generate a plot of pre-calibrated versus post-calibrated LRs using Pool Adjacent Violators (PAV).

    Parameters
    ----------
    ax : Axes
        The matplotlib axes object to plot on.
    llrdata : LLRData
        The LLRData object containing likelihood ratios and labels.
    add_misleading : int, optional
        Number of misleading evidence points to add on both sides (default: ``0``).
    show_scatter : bool, optional
        If `True`, show individual LRs (default: ``True``).
    """
    llrs = llrdata.llrs
    y = llrdata.hypothesis_labels

    pav = IsotonicCalibrator(add_misleading=add_misleading)
    pav_llrs = pav.fit_apply(llrdata).llrs

    xrange = yrange = [
        llrs[llrs != -np.inf].min() - 0.5,
        llrs[llrs != np.inf].max() + 0.5,
    ]

    legend = ax.get_legend()
    has_legend = legend is not None
    if not has_legend or (
        legend is not None and 'Consistent system' not in [text.get_text() for text in legend.get_texts()]
    ):
        ax.plot(xrange, yrange, '--', color='gray', label='Consistent system')

    # line pre pav llrs x and post pav llrs y
    line_x = np.arange(xrange[0], xrange[1], 0.01)
    line_y = pav.apply(LLRData(features=line_x.reshape(-1, 1))).llrs

    # filter nan values, happens when values are out of bound (x_values out of training domain for pav)
    # see: https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html
    line_x, line_y = line_x[~np.isnan(line_y)], line_y[~np.isnan(line_y)]

    # some values of line_y go beyond the yrange which is problematic when there are infinite values
    mask_out_of_range = np.logical_and(line_x >= yrange[0], line_x <= yrange[1])
    ax.plot(line_x[mask_out_of_range], line_y[mask_out_of_range], label='PAV transform', linewidth=2)

    # add points for infinite values
    if np.logical_or(np.isinf(pav_llrs), np.isinf(llrs)).any():

        def adjust_ticks_labels_and_range(
            neg_inf: bool, pos_inf: bool, axis_lower: float, axis_upper: float
        ) -> tuple[tuple[float, float], list[float], list[str]]:
            # We want a maximum of 10 ticks on the axis. If the range is larger,
            # we adjust the step_size accordingly. We devide by 9 because the range
            # is inclusive, so 10 ticks means 9 steps.
            step_size = (axis_upper - axis_lower) / 9

            ticks: list[int | float] = list(range(floor(axis_lower), ceil(axis_upper) + 1, ceil(step_size)))
            tick_labels = list(map(str, ticks))
            step_size_actual: int | float = ticks[2] - ticks[1]

            if neg_inf:
                axis_lower -= step_size_actual
                ticks = [axis_lower] + ticks
                tick_labels = ['-∞'] + tick_labels

            if pos_inf:
                axis_upper += step_size_actual
                ticks = ticks + [axis_upper]
                tick_labels = tick_labels + ['+∞']

            return (axis_lower, axis_upper), ticks, tick_labels

        def replace_values_out_of_range(values: np.ndarray, min_range: float, max_range: float) -> np.ndarray:
            # create margin for point so no overlap with axis line
            margin = (max_range - min_range) / 60
            return np.concatenate(
                [
                    np.where(np.isneginf(values), min_range + margin, values),
                    np.where(np.isposinf(values), max_range - margin, values),
                ]
            )

        yrange_tuple, ticks_y, tick_labels_y = adjust_ticks_labels_and_range(
            bool(np.isneginf(pav_llrs).any()), bool(np.isposinf(pav_llrs).any()), yrange[0], yrange[1]
        )
        xrange_tuple, ticks_x, tick_labels_x = adjust_ticks_labels_and_range(
            bool(np.isneginf(llrs).any()), bool(np.isposinf(llrs).any()), xrange[0], xrange[1]
        )

        mask_not_inf = np.logical_or(np.isinf(llrs), np.isinf(pav_llrs))
        x_inf = replace_values_out_of_range(llrs[mask_not_inf], xrange_tuple[0], xrange_tuple[1])
        y_inf = replace_values_out_of_range(pav_llrs[mask_not_inf], yrange_tuple[0], yrange_tuple[1])

        ax.set_yticks(ticks_y, tick_labels_y)
        ax.set_xticks(ticks_x, tick_labels_x)

        # Create lines at x=0 and y=0 to indicate the quadrants.
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle=':')
        ax.axvline(x=0, color='gray', linewidth=0.5, linestyle=':')

        color = [H1_COLOR if i > 0 else H2_COLOR for i in y_inf]
        ax.scatter(x_inf, y_inf, color=color, marker='|', linewidths=0.2)

        xrange = list(xrange_tuple)
        yrange = list(yrange_tuple)

    ax.axis((xrange[0], xrange[1], yrange[0], yrange[1]))
    # pre-/post-calibrated lr fit

    if show_scatter:
        mask_y = y == 1
        mask_not_y = ~mask_y

        h1_llrs = np.where(mask_y, llrs, np.nan)
        h1_pav = np.where(mask_y, pav_llrs, np.nan)
        h2_llrs = np.where(mask_not_y, llrs, np.nan)
        h2_pav = np.where(mask_not_y, pav_llrs, np.nan)

        y = check_not_none(y)
        n_h1 = np.count_nonzero(y)
        n_h2 = len(y) - n_h1

        ax.scatter(h1_llrs, h1_pav, facecolors=H1_COLOR, marker='v', linewidths=0.2, label=f'H1 (n={n_h1})')
        ax.scatter(h2_llrs, h2_pav, facecolors=H2_COLOR, marker='^', linewidths=0.2, label=f'H2 (n={n_h2})')

        # scatter plot of measured lrs

    ax.set_xlabel('pre-calibrated log$_{10}$(LR)')
    ax.set_ylabel('post-calibrated log$_{10}$(LR)')
    ax.legend()


def histogram(
    ax: Axes,
    x: np.ndarray,
    labels: np.ndarray | None,
    bins: int = 20,
    weighted: bool = True,
    x_label: str = '',
) -> None:
    """
    Plot x as a histogram, optionally separated by class labels.

    This class is mainly used as a helper for plotting LLR or score histograms.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes object to plot on.
    x : np.ndarray
        The array of values to plot.
    labels : np.ndarray | None
        The array of class labels for each value in x.
    bins : int
        Number of bins to divide scores into (default: 20).
    weighted : bool
        If y-axis should be weighted for frequency within each class (default: `True`).
    x_label : str
        Label for the x-axis (default: '').
    """
    bins_array = np.histogram_bin_edges(x[np.isfinite(x)], bins=bins).tolist()
    if labels is not None:
        points0, points1 = util.Xy_to_Xn(x, labels)
        weights0, weights1 = (np.ones_like(points) / len(points) if weighted else None for points in (points0, points1))
        ax.hist(points1, bins_array, alpha=0.25, weights=weights1, label=f'H1 (n={len(points1)})', color=H1_COLOR)
        ax.hist(points0, bins_array, alpha=0.25, weights=weights0, label=f'H2 (n={len(points0)})', color=H2_COLOR)
    else:
        weights = np.ones_like(x) / len(x) if weighted else None
        ax.hist(x, bins=bins_array, alpha=0.25, weights=weights, label=f'All data (n={len(x)})', color='gray')
    ax.set_xlabel(x_label)
    ax.set_ylabel('count' if not weighted else 'relative frequency')
    ax.legend()


def lr_histogram(
    ax: Axes,
    llrdata: LLRData,
    bins: int = 20,
    weighted: bool = True,
) -> None:
    """
    Plot the 10log LRs.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes object to plot on.
    llrdata : LLRData
        The LLRData object containing likelihood ratios and labels.
    bins : int
        Number of bins to divide scores into (default: 20).
    weighted : bool
        If y-axis should be weighted for frequency within each class (default: `True`).
    """
    histogram(ax, llrdata.llrs, llrdata.require_labels, bins=bins, weighted=weighted, x_label='log$_{10}$(LR)')


def tippett(ax: Axes, llrdata: LLRData, plot_type: int = 1) -> None:
    """
    Plot empirical cumulative distribution functions of same-source and different-sources LRs.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes object to plot on.
    llrdata : LLRData
        The LLRData object containing likelihood ratios and labels.
    plot_type : int
        Must be either 1 or 2 (default: 1).
        In type 1 both curves show proportion of lrs greater than or equal to the
        x-axis value, while in type 2 the curve for same-source shows the
        proportion of lrs smaller than or equal to the x-axis value.
    """
    llrs = llrdata.llrs
    labels = llrdata.require_labels

    lr_0, lr_1 = util.Xy_to_Xn(llrs, labels)
    finite_llrs = llrs[np.isfinite(llrs)]
    xvalues = np.linspace(np.min(finite_llrs), np.max(finite_llrs), 100)
    perc0 = (sum(i >= xvalues for i in lr_0) / len(lr_0)) * 100
    if plot_type == 1:
        perc1 = (sum(i >= xvalues for i in lr_1) / len(lr_1)) * 100
    elif plot_type == 2:
        perc1 = (sum(i <= xvalues for i in lr_1) / len(lr_1)) * 100
    else:
        raise ValueError(f'Argument plot_type in tippett() must be either 1 or 2, got `{plot_type}`.')
    ax.plot(xvalues, perc1, color='b', label=r'LRs given $\mathregular{H_1}$')
    ax.plot(xvalues, perc0, color='r', label=r'LRs given $\mathregular{H_2}$')
    ax.axvline(x=0, color='k', linestyle='--')
    ax.set_xlabel('log$_{10}$(LR)')
    ax.set_ylabel('Cumulative proportion')
    ax.legend()


def llr_interval(ax: Axes, llrdata: LLRData) -> None:
    """
    Plot the LRs on the x-axis, with the relative interval score on the y-axis.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes object to plot on.
    llrdata : LLRData
        The LLRData object containing the likelihood ratios and interval scores.
    """
    if not llrdata.has_intervals:
        raise ValueError('LLRData must contain interval scores to plot Score-LR.')

    llr_data = llrdata.features
    llr_sorted = np.sort(llr_data, axis=0)
    llrs = llr_sorted[:, 0]
    interval_scores_low = llr_sorted[:, 1] - llrs
    interval_scores_high = llr_sorted[:, 2] - llrs

    ax.plot(llrs, interval_scores_high, '|-', linewidth=0.5, color=H1_COLOR, label='Upper interval')
    ax.plot(llrs, interval_scores_low, '|-', linewidth=0.5, color=H2_COLOR, label='Lower interval')

    ax.axhline(y=0, color='gray', linestyle='--')

    ax.set_xlabel('Likelihood ratio (log$_{10}$)')
    ax.set_ylabel('Interval around LR (log$_{10}$(LR))')
    ax.legend(loc=1)


def score_distribution(
    ax: Axes,
    llrdata: LLRData,
    bins: int = 20,
    weighted: bool = True,
) -> None:
    """
    Plot the distributions of scores calculated by the (fitted) LR system.

    If `weighted` is `True`, the y-axis represents the probability density within the class. Otherwise, they-axis shows
    the number of instances.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes object to plot on.
    llrdata : LLRData
        The LLRData object containing the scores and labels. Must have scores available.
    bins : int
        Number of bins to divide scores into (default: 20).
    weighted : bool
        If y-axis should be the probability density within each class,
        instead of counts (default: `True`).
    """
    scores = llrdata.require_feature_for_plots('score')
    histogram(ax, scores, llrdata.require_labels, bins=bins, weighted=weighted, x_label='score')


def score_to_llr(ax: Axes, llrdata: LLRData) -> None:
    """Plot intermediate scores vs final LLRs, colored by hypothesis.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes object to plot on.
    llrdata : LLRData
        The LLRData object containing likelihood ratios and labels. Must have scores available. If labels are present,
        use them in the plots to color the points by hypothesis. If not, plot all points in the same color.
    """
    llrs = llrdata.llrs
    labels = llrdata.hypothesis_labels
    scores = llrdata.require_feature_for_plots('score')

    # General settings; do them regardless of labels.
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Intermediate Score')
    ax.set_ylabel('log$_{10}$(LR)')

    # If no labels, plot all points in the same color and return.
    if labels is None:
        ax.scatter(scores, llrs, alpha=0.5, label='Data points', color='gray')
        ax.legend()

        return

    mask_h1 = labels == 1
    mask_h2 = labels == 0

    n_h1 = np.count_nonzero(mask_h1)
    n_h2 = np.count_nonzero(mask_h2)

    ax.scatter(scores[mask_h1], llrs[mask_h1], alpha=0.5, label=f'H1 (n={n_h1})', color=H1_COLOR)
    ax.scatter(scores[mask_h2], llrs[mask_h2], alpha=0.5, label=f'H2 (n={n_h2})', color=H2_COLOR)
    ax.legend()
