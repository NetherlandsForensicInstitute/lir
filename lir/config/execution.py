import itertools
import logging
from collections.abc import Iterable
from copy import deepcopy
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
    """Base class for LR system and data configurations."""

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
