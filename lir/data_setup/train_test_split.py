import functools
from collections.abc import Iterable
from pathlib import Path

from lir import DataProvider, DataSetup, DataStrategy, InstanceData, Transformer
from lir.config import ConfigAttribute, ConfigValue, config_parser
from lir.config.data import parse_data_provider, parse_splitting_strategy
from lir.config.transform import parse_module
from lir.transform import Identity


class SplitDataSetup(DataSetup):
    """
    A data setup that uses a single data provider to construct pairs of training and test sets.

    A split data setup consists of three components: a data provider, a filter, and a strategy.

    The filter is a :class:`~lir.Transformer` that supports calling the `apply()` method without first calling
    `fit()`. Unlike in LR system pipelines, this transformer may change the number of instances in the dataset.

    Parameters
    ----------
    provider : DataProvider
        The :class:`~lir.data.models.DataProvider` that retrieves the data from some data source, such as
          a CSV file or a database.
    strategy : DataStrategy
        The :class:`~lir.data.models.DataStrategy` that determines how the data are used.
    data_filter : Transformer | None
        An optional filter (:class:`~lir.Transformer`) to apply to the raw data before doing anything else.
    use_cache : bool, optional
        If ``True``, the train/inference pairs will be cached locally. Defaults to ``False``.
    """

    def __init__(
        self, provider: DataProvider, strategy: DataStrategy, data_filter: Transformer | None, use_cache: bool = False
    ):
        self.provider = provider
        self.strategy = strategy
        self.filter = data_filter or Identity()

        self._get_train_inference_pairs_cached = (
            functools.cache(self._get_train_inference_pairs) if use_cache else self._get_train_inference_pairs
        )

    def _get_train_inference_pairs(self) -> Iterable[tuple[InstanceData, InstanceData]]:
        """
        Return the data in the form of one or more train/test splits.

        This method follows three steps:
        - retrieve instances from the data provider;
        - pass them through the filter by calling its `apply()` method;
        - apply the data strategy to arrange them into one or more train/test splits.

        This method caches results for speeding up consecutive calls.

        Returns
        -------
        Iterable[tuple[InstanceData, InstanceData]]
            An iterator over tuples of train/test splits.
        """
        return list(self.strategy.apply(self.filter.apply(self.provider.get_instances())))

    def get_train_inference_pairs(self) -> Iterable[tuple[InstanceData, InstanceData]]:
        """
        Return the data in the form of one or more train/test splits.

        This method follows three steps:
        - retrieve instances from the data provider;
        - pass them through the filter by calling its `apply()` method;
        - apply the data strategy to arrange them into one or more train/test splits.

        This method caches results for speeding up consecutive calls if the class is instantiated with
        ``use_cache=True``.

        Returns
        -------
        Iterable[tuple[InstanceData, InstanceData]]
            An iterator over tuples of train/test splits.
        """
        return self._get_train_inference_pairs_cached()


@config_parser(
    attributes=[
        ConfigAttribute('provider', DataProvider, required=True),
        ConfigAttribute('filter', Transformer, required=False),
        ConfigAttribute('splits', DataStrategy, required=True),
        ConfigAttribute('use_cache', bool, default=False),
    ]
)
def parse(config: ConfigValue, output_path: Path) -> SplitDataSetup:
    """
    Parse data provider and data strategy from configuration.

    The fields `provider`, `filter` and `splits` are parsed, which are expected to refer
    to specific implementations of `DataProvider`, `Transformer` and `DataStrategy`, respectively.
    See `parse_data_provider`, `parse_module` and `parse_data_strategy` for more information.

    Parameters
    ----------
    config : ConfigValue
        Configuration section containing provider and split strategy.
    output_path : Path
        Output path for created objects.

    Returns
    -------
    DataSetup
        Parsed data provider, filter and strategy.
    """
    with config:
        provider = parse_data_provider(config.pop('provider'), output_path)
        data_filter = parse_module(config.pop('filter', required=False), output_path)
        strategy = parse_splitting_strategy(config.pop('splits'), output_path)
        use_cache = config.pop_field('use_cache', default=False)
        return SplitDataSetup(provider, strategy, data_filter, use_cache=use_cache)
