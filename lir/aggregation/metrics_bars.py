import logging
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from types import NoneType
from typing import IO, Any

import numpy as np
from matplotlib import pyplot as plt

from lir import LLRData
from lir.aggregation.base import Aggregation, AggregationData
from lir.config.base import ConfigValue, YamlParseError, check_is_empty, config_parser, pop_field
from lir.config.metrics import parse_individual_metric


LOG = logging.getLogger(__name__)


class MetricsBarPlot(Aggregation):
    """
    Generate a bar plot for metrics and runs.

    The plot shows a bar for each LR-system run and metric combination.

    Usage example in YAML:

    .. code-block:: yaml

        output:
          - method: metric_bars
            plot_params:
              ylim: [None, 1]
              xlabel: run
          - metrics:
            - cllr
            - cllr_min

    .. jupyter-execute::
        :hide-code:

        from lir.aggregation import MetricsBarPlot, AggregationData
        from lir import metrics, LLRData
        import numpy as np

        results = [
            AggregationData(
                run_name='1',
                llrdata=LLRData(
                    features=np.array([9., 9, 9, .5, .5, -9, -9, -9]).reshape(-1, 1),
                    hypothesis=np.array([1, 1, 1, 1, 0, 0, 0, 0])
                ),
                parameters={'model': 'model1'},
                lrsystem=None,
                experiment_output_dir='results',
                run_output_dir='results/run',
            ),
            AggregationData(
                run_name='1',
                llrdata=LLRData(
                    features=np.array([9., 9, 9, 9, 9, .5, .5, -9, -9, -9, -9, -9]).reshape(-1, 1),
                    hypothesis=np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
                ),
                parameters={'model': 'model2'},
                lrsystem=None,
                experiment_output_dir='results',
                run_output_dir='results/run',
            ),
        ]

        aggr = MetricsBarPlot(
            path=None,
            metrics={'cllr': metrics.cllr, 'cllr_min': metrics.cllr_min},
            plot_params={'ylim': [None, 1], 'xlabel': 'run'}
        )
        for data in results:
            aggr.report(data)
        aggr.close()

    Parameters
    ----------
    path : Path | None
        The path to where the plot file is written.
    metrics : Mapping[str, Callable]
        A mapping of metric names to functions that compute the values for the metrics.
    plot_params : Mapping[str, Any]
        Properties of the plot, passed as keyword arguments to :meth:`matplotlib.axes.Axes.update`.
    """

    def __init__(
        self,
        path: Path | None,
        metrics: Mapping[str, Callable[[LLRData], float | list[float]]],
        plot_params: dict[str, Any] | None = None,
    ):
        self.path = path
        self.full_path: Path | None = None
        self._file: IO[Any] | None = None
        self.metric_functions = OrderedDict(metrics.items())
        self.calculated_values: list[list[float | None]] = []
        self.run_names: list[str] = []
        self.plot_params: dict[str, Any] = plot_params or {}

    @staticmethod
    def _safe_call(fn: Callable, message: str) -> float | None:
        try:
            value = fn()
            if not isinstance(value, (float, NoneType)):
                raise ValueError(f'not a number: {value}')
            return value
        except Exception as e:
            LOG.warning(f'{message}: {e}')
            return None

    def report(self, data: AggregationData) -> None:
        """
        Write the metrics to a plot.

        Parameters
        ----------
        data : AggregationData
            The data for which to compute metrics.
        """
        self.calculated_values.append(
            [
                (self._safe_call(partial(fn, data.llrdata), f'calculating metric {key} failed'))
                for key, fn in self.metric_functions.items()
            ]
        )

        self.run_names.append(', '.join(str(opt) for opt in data.parameters.values()))

        if self.path is not None:
            self.full_path = data.resolve_path_for_experiment(self.path)

    def close(self) -> None:
        """Ensure the CSV file is properly closed after writing."""
        fig, ax = plt.subplots()
        metric_names = list(self.metric_functions.keys())
        n_runs = len(self.run_names)
        n_metrics = len(metric_names)
        run_margin = 0.5
        run_width = n_metrics + run_margin

        for metric_index in range(n_metrics):
            x_values = np.arange(n_runs) * run_width + metric_index
            metric_values = np.array([run_values[metric_index] for run_values in self.calculated_values])
            ax.bar(x_values, metric_values, label=metric_names[metric_index], width=1.0, align='edge', alpha=0.5)

        ax.set_xticks(np.arange(n_runs) * run_width + n_metrics / 2, self.run_names, rotation=20, ha='right')
        ax.update(self.plot_params)

        fig.legend()
        fig.tight_layout()
        if self.full_path:
            fig.savefig(self.full_path)
        else:
            plt.show(block=True)
        plt.close(fig)


@config_parser
def parse(config: ConfigValue, output_dir: Path) -> MetricsBarPlot:
    """
    Parse a configuration section into an :class:`~lir.aggregation.metrics_bars.MetricsBarPlot`.

    Parameters
    ----------
    config : ConfigValue
        The configuration dictionary.
    output_dir : Path
        The directory where the plot will be saved.

    Returns
    -------
    MetricsBarPlot
        An instance of :class:`~lir.aggregation.metrics_bars.MetricsBarPlot`.
    """
    metric_names = pop_field(config, 'metrics', default=['cllr'])
    if not isinstance(metric_names, Sequence):
        raise YamlParseError(
            config.context,
            'Invalid metrics configuration; expected a list of metric names.',
        )

    path: Path = pop_field(config, 'path', default=Path('metrics.png'), validate=Path)
    metrics = {name: parse_individual_metric(name, output_dir, config.context) for name in metric_names}
    plot_params = pop_field(config, 'plot_params', default={})

    check_is_empty(config)
    return MetricsBarPlot(path, metrics, plot_params=plot_params)
