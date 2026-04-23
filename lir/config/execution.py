import itertools
import logging
import math
import multiprocessing
import os
from collections.abc import Callable, Iterable
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import confidence

from lir import InstanceData, LLRData
from lir.aggregation import AggregationData
from lir.config.base import ContextAwareDict
from lir.config.data import parse_data_object
from lir.config.lrsystem_architectures import parse_lrsystem
from lir.config.util import simplify_data_structure
from lir.data.models import DataProvider, DataStrategy, concatenate_instances
from lir.lrsystems import LRSystem


LOG = logging.getLogger(__name__)


class BaseConfig(NamedTuple):
    """
    Base class for LR system and data configurations.

    A configuration of an LR system or data setup is a dictionary: the ``spec`` attribute. Additionally, there can be
    hyperparameters that are already incorporated in the configuration and can be used to describe the configuration in
    how it is different from other configurations.

    The configuration is extended by the subclass to lazily materialize the configuration on demand.

    Objects of this class are pickleable. When pickled, the materialization is dropped and will have to be recreated
    when needed.

    Attributes
    ----------
    spec : ContextAwareDict
        The configuration of an LR system or data setup for a run.
    params : dict[str, Any]
        The parameters that describe the configuration.
    output_dir : Path
        Path to the directory where results may be written. As this directory is shared among all runs of an experiment,
        a dedicated subdirectory may be created for the run.
    """

    spec: ContextAwareDict
    params: dict[str, Any]
    output_dir: Path

    @property
    def desc(self) -> str:
        """
        Generate a description of this configuration from the parameter values.

        Returns
        -------
        str
            A description of this configuration.
        """
        return '__'.join([f'{key}={value}' for key, value in self.params.items()])

    def __reduce__(self) -> tuple[Callable, tuple]:
        return self.__class__, (self.spec, self.params, self.output_dir)


class DataConfig(BaseConfig):
    """Data configuration object."""

    _data_object = None
    _splits = None

    def _parse_data_object(self) -> tuple[DataProvider, DataStrategy]:
        """
        Parse the data configuration.

        This is done here to ensure that data
        parsing is only done once per data configuration, even when multiple
        LR systems are being evaluated on the same data setup.

        Returns
        -------
        tuple[DataProvider, DataStrategy]
            A tuple of a data provider and a data strategy.
        """
        if self._data_object is None:
            self._data_object = parse_data_object(deepcopy(self.spec), self.output_dir)
        return self._data_object

    @property
    def provider(self) -> DataProvider:
        """
        Return a data provider.

        Returns
        -------
        DataProvider
            A data provider object.
        """
        return self._parse_data_object()[0]

    @property
    def splitter(self) -> DataStrategy:
        """
        Return a data splitter.

        Returns
        -------
        DataStrategy
            A data splitter object.
        """
        return self._parse_data_object()[1]

    def splits(self) -> Iterable[tuple[InstanceData, InstanceData]]:
        """
        Convert the split_data iterable to a list to allow multiple iterations over the splits.

        E.g. one iteration for validation and one for case LLR generation.

        Returns
        -------
        Iterable[tuple[InstanceData, InstanceData]]
            An iterable of training/test set pairs.
        """
        if self._splits is None:
            self._splits = list(self.splitter.apply(self.provider.get_instances()))
        return self._splits


class LRSystemConfig(BaseConfig):
    """LR system configuration object."""

    _lrsystem = None

    @property
    def lrsystem(self) -> LRSystem:
        """
        Return the materialized LR system as defined in configuration.

        Returns
        -------
        LRSystem
            An LR system object.
        """
        if self._lrsystem is None:
            self._lrsystem = parse_lrsystem(deepcopy(self.spec), self.output_dir)
        return self._lrsystem


def run_lrsystem(
    output_base_dir: Path,
    lrsystem_config: LRSystemConfig,
    data_config: DataConfig,
    skip_full_lrsystem: bool = False,
    run_name: str | None = None,
) -> AggregationData:
    """
    Run experiment on a single LR system configuration using the provided data.

    The LR system is fitted using the training subset data and
    subsequently used to determine LLRs for the test subset data. The results are stored in
    a temporary list which contains the determined data of each test / train split.

    The collected results are combined and passed to the configured `outputs` aggregations,
    which may write metrics and visualizations to the `output_path` directory. The combined
    LLR data is returned.

    Next to this, the configuration of both the data and LR system are stored in the output
    directory for future reference.

    Parameters
    ----------
    output_base_dir : Path
        The base directory of the path where results may be written.
    lrsystem_config : LRSystemConfig
        LR-system configuration for a single run.
    data_config : DataConfig
        Data configuration used to construct datasets for runs.
    skip_full_lrsystem : bool
        If True, the full LR system will not be trained.
    run_name : str | None
        The name of the run (optional). If None, the name will be derived from parameter values.

    Returns
    -------
    LLRData
        Likelihood-ratio data produced by applying the LR system.
    """
    # ombine the data parameter with the LR system parameters to create a unique name for this run.
    run_name: str = (
        run_name
        or f'{data_config.desc}{"__" if lrsystem_config.desc and data_config.desc else ""}{lrsystem_config.desc}'
    )

    output_dir = output_base_dir / run_name if run_name else output_base_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    # This dictionary contains all parameters for this experiment run, prefixed
    # by either 'data.' or 'lrsystem.' to avoid naming conflicts. This is used
    # for reporting purposes in the aggregations.
    parameters = {f'data.{k}': v for k, v in data_config.params.items()} | {
        f'lrsystem.{k}': v for k, v in lrsystem_config.params.items()
    }

    # Turn the configurations into dictionaries for writing to YAML files.
    lrsystem_config_dict = simplify_data_structure(lrsystem_config.spec)
    data_config_dict = simplify_data_structure(data_config.spec)

    # Check that simplify_data_structure returned a dict for type checking.
    if not isinstance(lrsystem_config_dict, dict) or not isinstance(data_config_dict, dict):
        raise ValueError('hyperparameters are not the expected type (dict)')

    confidence.dumpf(confidence.Configuration(lrsystem_config_dict), output_dir / 'lrsystem.yaml')
    confidence.dumpf(confidence.Configuration(data_config_dict), output_dir / 'data.yaml')

    # Placeholders for numpy arrays of LLRs and labels obtained from each train/test split
    llrs: list[LLRData] = []

    for training_data, test_data in data_config.splits():
        lrsystem_config.lrsystem.fit(training_data)
        llrs.append(lrsystem_config.lrsystem.apply(test_data))

    # Combine collected numpy arrays after iteration over the train/test split(s)
    llrs: LLRData = concatenate_instances(*llrs)

    # Create a lazy factory for full-data-fitted model with memoization
    _cached_full_fit_lrsystem = None

    def get_full_fit_lrsystem() -> LRSystem:
        nonlocal _cached_full_fit_lrsystem
        if _cached_full_fit_lrsystem is None:
            full_training_data = concatenate_instances(*next(iter(data_config.splits())))
            _cached_full_fit_lrsystem = parse_lrsystem(deepcopy(lrsystem_config.spec), output_dir)
            _cached_full_fit_lrsystem.fit(full_training_data)
        return _cached_full_fit_lrsystem

    # Collect and report results as configured by `outputs`
    return AggregationData(
        llrdata=llrs,
        lrsystem=lrsystem_config.lrsystem,
        parameters=parameters,
        run_name=run_name,
        get_full_fit_lrsystem=None if skip_full_lrsystem else get_full_fit_lrsystem,
    )


def run_multiple(
    output_base_dir: Path, lrsystem_configs: list[LRSystemConfig], data_configs: list[DataConfig]
) -> Iterable[AggregationData]:
    """
    Run LR systems sequentially.

    Consider using :meth:`parallellize_runs` to speed up processing by doing runs in parallel.

    Parameters
    ----------
    output_base_dir : Path
        The base directory where the results may be written.
    lrsystem_configs : list[LRSystemConfig]
        A list of LR system configuraitons.
    data_configs : list[DataConfig]
        A list of dataset configurations.

    Returns
    -------
    list[AggregationData]
        A list of results for all runs.
    """
    LOG.debug(f'doing {len(lrsystem_configs) * len(data_configs)} runs sequentially')
    for lrsystem_config, data_config in itertools.product(lrsystem_configs, data_configs):
        yield run_lrsystem(output_base_dir, lrsystem_config, data_config)


def run_multiple_lrsystems(
    output_base_dir: Path, lrsystem_configs: list[LRSystemConfig], data_config: DataConfig
) -> list[AggregationData]:
    """
    Run multiple LR systems for a single data configuration.

    Parameters
    ----------
    output_base_dir : Path
        The base directory where the results may be written.
    lrsystem_configs : list[LRSystemConfig]
        A list of LR system configuraitons.
    data_config : DataConfig
        Data configuration used to construct the dataset.

    Returns
    -------
    list[AggregationData]
        A list of results for all runs.
    """
    LOG.debug(f'process {multiprocessing.current_process()} about to do {len(lrsystem_configs)} runs')
    try:
        return [
            run_lrsystem(output_base_dir, lrsystem_config, data_config, skip_full_lrsystem=True)
            for lrsystem_config in lrsystem_configs
        ]
    finally:
        LOG.debug(f'process {multiprocessing.current_process()} finished {len(lrsystem_configs)} runs')


def parallellize_runs(
    output_base_dir: Path, lrsystem_configs: list[LRSystemConfig], data_configs: list[DataConfig]
) -> Iterable[AggregationData]:
    """
    Run LR systems in parallel.

    This method has exactly the same effect as :meth:`run_multiple`, but uses ``multiprocessing`` to do runs in
    parallel. It selects a parallelization strategy to distribute the runs over workers.

    Issues:

    - this may lead to repetitive loading of data, which may take additional (costly) I/O operations
    - in some cases (notably, when bootstrapping) the `multiprocessing.imap_unsorted` operation may produce a "leaked
      semaphore" warning
    - logging in workers is disabled

    Parameters
    ----------
    output_base_dir : Path
        The base directory where the results may be written.
    lrsystem_configs : list[LRSystemConfig]
        A list of LR system configuraitons.
    data_configs : list[DataConfig]
        A list of dataset configurations.

    Returns
    -------
    list[AggregationData]
        A list of results for all runs.
    """
    n_runs = len(lrsystem_configs) * len(data_configs)
    n_processes = os.process_cpu_count()

    if n_runs == 1 or n_processes == 1:
        # don't bother parallellizing if there is only a single configuration or a single CPU
        yield from run_multiple(output_base_dir, lrsystem_configs, data_configs)

    with multiprocessing.Pool(processes=n_processes) as pool:
        if len(data_configs) > 1:
            # there are multiple data setups --> iterate over data setups first recycles data loading

            LOG.debug(f'spawning {len(data_configs)} tasks to do a total of {n_runs} runs ')
            for results in pool.imap(partial(run_multiple_lrsystems, output_base_dir, lrsystem_configs), data_configs):
                yield from results
        else:
            # there is a single data setup and multiple lrsystems --> iterate over lrsystems

            # we need no more chunks that the number of processes
            chunksize = math.ceil(len(lrsystem_configs) / n_processes)

            LOG.debug(f'spawning {len(lrsystem_configs)} tasks to do a total of {n_runs} runs in chunks of {chunksize}')
            yield from pool.imap_unordered(
                partial(run_lrsystem, output_base_dir, data_config=data_configs[0], skip_full_lrsystem=True),
                lrsystem_configs,
                chunksize=chunksize,
            )
