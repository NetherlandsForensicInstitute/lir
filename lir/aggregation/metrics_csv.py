import csv
import logging
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import IO, Any

from lir.aggregation.base import Aggregation, AggregationData
from lir.config.base import ContextAwareDict, YamlParseError, check_is_empty, config_parser, pop_field
from lir.config.metrics import parse_individual_metric


LOG = logging.getLogger(__name__)


class WriteMetricsToCsv(Aggregation):
    """
    Helper class to write aggregated results to CSV file.

    Parameters
    ----------
    path : Path
        The path to the CSV file where the metrics will be written.
    columns : Mapping[str, Callable]
        A mapping of column names to metric functions that compute the values for those columns.
    """

    def __init__(self, path: Path | str, columns: Mapping[str, Callable]):
        self.path = Path(path)
        self._file: IO[Any] | None = None
        self._writer: csv.DictWriter | None = None
        self.columns = columns

    @staticmethod
    def _safe_call(fn: Callable, message: str) -> Any:
        try:
            return fn()
        except Exception as e:
            LOG.warning(f'{message}: {e}')
            return ''

    def report(self, data: AggregationData) -> None:
        """
        Write the metrics to CSV.

        Parameters
        ----------
        data : AggregationData
            The aggregated data for which to compute and write the metrics.
        """
        columns = [
            (key, self._safe_call(partial(metric, data.llrdata), f'calculating metric {key} failed'))
            for key, metric in self.columns.items()
        ]
        metrics = []
        for name, value in columns:
            if isinstance(value, (list, tuple)):
                for index, metric_value in enumerate(value):
                    metrics.append((f'{name}_{index}', str(metric_value)))
            else:
                metrics.append((name, str(value)))

        results = OrderedDict([(k, str(v)) for k, v in data.parameters.items()] + metrics)

        # Record column header names only once to the CSV
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(data.resolve_path_for_experiment(self.path), 'w', newline='')  # noqa: SIM115
            self._writer = csv.DictWriter(self._file, fieldnames=results.keys())
            self._writer.writeheader()
        self._writer.writerow(results)
        self._file.flush()  # type: ignore

    def close(self) -> None:
        """Ensure the CSV file is properly closed after writing."""
        if self._file:
            self._file.close()


@config_parser
def parse(config: ContextAwareDict, output_dir: Path) -> WriteMetricsToCsv:
    """
    Corresponding registry function to leverage CSV Writer class to write results to disk.

    Parameters
    ----------
    config : ContextAwareDict
        The configuration dictionary for the metrics CSV.
    output_dir : Path
        The directory where the metrics CSV will be saved.

    Returns
    -------
    WriteMetricsToCsv
        An instance of the WriteMetricsToCsv class configured to write metrics to a CSV file.
    """
    columns = pop_field(config, 'columns', default=['cllr', 'cllr_min'])
    if not isinstance(columns, Sequence):
        raise YamlParseError(
            config.context,
            'Invalid metrics configuration; expected a list of metric names.',
        )

    columns = {name: parse_individual_metric(name, output_dir, config.context) for name in columns}

    check_is_empty(config)
    return WriteMetricsToCsv('metrics.csv', columns)
