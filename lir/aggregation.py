import csv
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import IO, Any, NamedTuple

import numpy as np
from matplotlib import pyplot as plt

from lir.algorithms.bayeserror import plot_nbe as nbe
from lir.algorithms.invariance_bounds import plot_invariance_delta_functions as invariance_delta_functions
from lir.algorithms.llr_overestimation import plot_llr_overestimation as llr_overestimation
from lir.config.base import ContextAwareDict, YamlParseError, check_is_empty, config_parser, pop_field
from lir.config.data import parse_data_provider
from lir.config.metrics import parse_individual_metric
from lir.data.models import DataProvider, LLRData, get_instances_by_category
from lir.lrsystems.lrsystems import LRSystem
from lir.plotting import llr_interval, lr_histogram, pav, tippett
from lir.plotting.expected_calibration_error import plot_ece as ece


LOG = logging.getLogger(__name__)


class AggregationData(NamedTuple):
    """Representation of aggregated data.

    Fields:
    - llrdata: the LLR data containing LLRs and labels.
    - lrsystem: the model that produced the results
    - get_full_fit_lrsystem: optional callable that lazily provides a model fitted on full data (ignoring splits)
    - parameters: parameters that identify the system producing the results
    - run_name: string representation of the run that produced the results
    """

    llrdata: LLRData
    lrsystem: LRSystem
    parameters: dict[str, Any]
    run_name: str
    get_full_fit_lrsystem: Callable[[], LRSystem] | None = None


class Aggregation(ABC):
    """Base representation of an aggregated data collection.

    Other classes may extend from this class.
    """

    @abstractmethod
    def report(self, data: AggregationData) -> None:
        """
        Report that new results are available.

        :param data: a named tuple containing the results
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

    def __init__(self, plot_fn: Callable, plot_name: str, output_dir: Path | None = None, **kwargs: Any) -> None:
        self.output_path = output_dir
        self.plot_fn = plot_fn
        self.plot_name = plot_name
        self.plot_fn_args = kwargs

    def report(self, data: AggregationData) -> None:
        """Plot the data when new results are available."""
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


@config_parser
def plot_pav(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    """Corresponding registry function to generate aggregate PAV plot."""
    plot_name = pop_field(config, 'plot_name', default='PAV')
    return AggregatePlot(pav, plot_name, output_dir, **config)


@config_parser
def plot_ece(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    """Corresponding registry function to generate aggregate ECE plot."""
    plot_name = pop_field(config, 'plot_name', default='ECE')
    return AggregatePlot(ece, plot_name, output_dir, **config)


@config_parser
def plot_lr_histogram(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    """Corresponding registry function to generate aggregate LR Histogram."""
    plot_name = pop_field(config, 'plot_name', default='LR_Histogram')
    return AggregatePlot(lr_histogram, plot_name, output_dir, **config)


@config_parser
def plot_llr_interval(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    """Corresponding registry function to generate aggregate LLR interval plot."""
    plot_name = pop_field(config, 'plot_name', default='LLR_Interval')
    return AggregatePlot(llr_interval, plot_name, output_dir, **config)


@config_parser
def plot_llr_overestimation(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    """Corresponding registry function to generate aggregate LLR overestimation plot."""
    plot_name = pop_field(config, 'plot_name', default='LLR_Overestimation')
    return AggregatePlot(llr_overestimation, plot_name, output_dir, **config)


@config_parser
def plot_nbe(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    """Corresponding registry function to generate aggregate NBE plot."""
    plot_name = pop_field(config, 'plot_name', default='NBE')
    return AggregatePlot(nbe, plot_name, output_dir, **config)


@config_parser
def plot_tippett(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    """Corresponding registry function to generate aggregate Tippett plot."""
    plot_name = pop_field(config, 'plot_name', default='Tippett')
    return AggregatePlot(tippett, plot_name, output_dir, **config)


@config_parser
def plot_invariance_delta_function(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    """Corresponding registry function to generate aggregate invariance delta function plot."""
    plot_name = pop_field(config, 'plot_name', default='Invariance_Delta_Functions')
    return AggregatePlot(invariance_delta_functions, plot_name, output_dir, **config)


class WriteMetricsToCsv(Aggregation):
    """Helper class to write aggregated results to CSV file."""

    def __init__(self, path: Path, columns: Mapping[str, Callable]):
        """
        Initialize the class.

        :param path: the path to the CSV file
        :param columns: the columns as a dictionary of names to metric functions
        """
        self.path = path
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
        """Write the metrics to CSV."""
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

        results = OrderedDict(list(data.parameters.items()) + metrics)

        # Record column header names only once to the CSV
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, 'w', newline='')  # noqa: SIM115
            self._writer = csv.DictWriter(self._file, fieldnames=results.keys())
            self._writer.writeheader()
        self._writer.writerow(results)
        self._file.flush()  # type: ignore

    def close(self) -> None:
        """Ensure the CSV file is properly closed after writing."""
        if self._file:
            self._file.close()


@config_parser
def metrics_csv(config: ContextAwareDict, output_dir: Path) -> WriteMetricsToCsv:
    """Corresponding registry function to leverage CSV Writer class to write results to disk."""
    columns = pop_field(config, 'columns', default=['cllr', 'cllr_min'])
    if not isinstance(columns, Sequence):
        raise YamlParseError(
            config.context,
            'Invalid metrics configuration; expected a list of metric names.',
        )

    columns = {name: parse_individual_metric(name, output_dir, config.context) for name in columns}

    check_is_empty(config)
    return WriteMetricsToCsv(output_dir / 'metrics.csv', columns)


class CaseLLRToCsv(Aggregation):
    """Aggregation that applies a full-data-fitted LR system to case data and stores LLRs as CSV."""

    def __init__(self, output_dir: Path, case_data_provider: DataProvider, filename: str = 'case_llr.csv') -> None:
        self.output_dir = output_dir
        self.case_data_provider = case_data_provider
        self.filename = Path(filename)

    def _get_output_path(self, run_name: str) -> Path:
        if self.filename.is_absolute() or self.filename.is_relative_to(self.output_dir):
            return self.filename

        dirname = self.output_dir / run_name if run_name else self.output_dir
        dirname.mkdir(parents=True, exist_ok=True)
        return dirname / self.filename

    @staticmethod
    def _feature_columns(features: np.ndarray, header: list[str] | None = None) -> tuple[list[str], list[np.ndarray]]:
        if features.ndim < 2:
            raise ValueError(f'unsupported feature shape for CSV export: {features.shape}')

        features_2d = features.reshape(features.shape[0], -1)
        feature_count = features_2d.shape[1]

        if header is not None and len(header) == feature_count:
            column_names = [str(column) for column in header]
        else:
            column_names = [f'feature_{index}' for index in range(feature_count)]

        return column_names, [features_2d[:, i] for i in range(feature_count)]

    def report(self, data: AggregationData) -> None:
        """Apply the full-data-fitted LR system to the case data and store the resulting LLRs as CSV."""
        if data.get_full_fit_lrsystem is not None:
            lrsystem = data.get_full_fit_lrsystem()
        else:
            LOG.warning(
                f'No full-data-fitted model factory available for run `{data.run_name}`; '
                f'using split-trained model instead.'
            )
            lrsystem = data.lrsystem

        # Ensure the case data does not contain labels by setting them to None.
        case_instances = self.case_data_provider.get_instances().replace(labels=None)
        case_llrs = lrsystem.apply(case_instances)

        path = self._get_output_path(data.run_name)
        path.parent.mkdir(parents=True, exist_ok=True)

        feature_header = getattr(case_instances, 'header', None)
        feature_headers, feature_values = self._feature_columns(case_instances.features, feature_header)

        columns: list[tuple[str, np.ndarray]] = list(zip(feature_headers, feature_values, strict=True))
        columns.append(('llr', case_llrs.llrs))

        if case_llrs.has_intervals and case_llrs.llr_intervals is not None:
            columns.extend(
                [
                    ('llr_interval_low', case_llrs.llr_intervals[:, 0]),
                    ('llr_interval_high', case_llrs.llr_intervals[:, 1]),
                ]
            )

        if len(case_instances) != len(case_llrs):
            raise ValueError(
                f'Cannot export original case features to case_llr.csv because row counts differ: '
                f'{len(case_instances)} case rows vs {len(case_llrs)} LLR rows.'
            )

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name for name, _ in columns])
            for row in zip(*(values for _, values in columns), strict=True):
                writer.writerow(row)


@config_parser
def case_llr_csv(config: ContextAwareDict, output_dir: Path) -> CaseLLRToCsv:
    """Parse output configuration for case LLR generation and CSV export."""
    case_data_provider = parse_data_provider(pop_field(config, 'case_llr_data'), output_dir)
    filename = pop_field(config, 'filename', default='case_llr.csv', validate=str)
    check_is_empty(config)
    return CaseLLRToCsv(output_dir, case_data_provider, filename)


class SubsetAggregation(Aggregation):
    """
    Aggregation method that manages data categorization.

    A separate aggregation method is used for each category.
    """

    def __init__(self, aggregation_methods: list[Aggregation], category_field: str):
        """
        Initialize the subset aggregation method.

        :param aggregation_methods: a list of methods to aggregate results by category
        :param category_field: the name of the category field
        """
        self.aggregation_methods = aggregation_methods
        self.category_field = category_field

    def report(self, data: AggregationData) -> None:
        """
        Report that new results are available.

        The data are categorized into subsets and forwarded to the actual aggregation method.

        :param data: a named tuple containing the results
        """
        run_name_prefix = f'{data.run_name}/' if data.run_name else ''
        for category, subset in get_instances_by_category(data.llrdata, self.category_field):
            category_str = '_'.join(str(v) for v in category.reshape(-1))
            run_name = f'{run_name_prefix}{category_str}'
            category_data = AggregationData(
                llrdata=subset,
                lrsystem=data.lrsystem,
                parameters=data.parameters | {self.category_field: category},
                run_name=run_name,
                get_full_fit_lrsystem=data.get_full_fit_lrsystem,
            )

            for output in self.aggregation_methods:
                output.report(category_data)

    def close(self) -> None:
        """Close all subset aggregation methods."""
        for output in self.aggregation_methods:
            output.close()
