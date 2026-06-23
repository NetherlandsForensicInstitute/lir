import functools
from collections.abc import Callable, Iterable
from pathlib import Path

from lir import Transformer, registry
from lir.config.base import (
    GenericConfigParser,
    YamlParseError,
    check_is_empty,
    pop_field,
)
from lir.config.substitution import ContextAwareDict
from lir.config.transform import parse_module
from lir.data.models import DataProvider, DataStrategy, InstanceData
from lir.transform import Identity


class DataSetup:
    """
    Data setup, consisting of three components: a data provider, a filter, and a strategy.

    The filter is a :class:`~lir.Transformer` that supports calling the `apply()` method without priorly calling
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
    """

    def __init__(self, provider: DataProvider, strategy: DataStrategy, data_filter: Transformer | None):
        self.provider = provider
        self.strategy = strategy
        self.filter = data_filter or Identity()

    def get_splits(self) -> Iterable[tuple[InstanceData, InstanceData]]:
        """
        Return the data in the form of one or more train/test splits.

        This method follows three steps:
        - retrieve instances from the data provider;
        - pass them through the filter by calling its `apply()` method;
        - apply the data strategy to arrange them into one or more train/test splits.

        Returns
        -------
        Iterable[tuple[InstanceData, InstanceData]]
            An iterator over tuples of train/test splits.
        """
        return self.strategy.apply(self.filter.apply(self.provider.get_instances()))


def parse_data_setup(cfg: ContextAwareDict, output_path: Path) -> DataSetup:
    """
    Parse data provider and data strategy from configuration.

    The fields `provider`, `filter` and `splits` are parsed, which are expected to refer
    to specific implementations of `DataProvider`, `Transformer` and `DataStrategy`, respectively.
    See `parse_data_provider`, `parse_module` and `parse_data_strategy` for more information.

    Parameters
    ----------
    cfg : ContextAwareDict
        Configuration section containing provider and split strategy.
    output_path : Path
        Output path for created objects.

    Returns
    -------
    DataSetup
        Parsed data provider, filter and strategy.
    """
    provider = parse_data_provider(pop_field(cfg, 'provider'), output_path)
    data_filter = parse_module(pop_field(cfg, 'filter', required=False), output_path, cfg.context + ['filter'])
    strategy = parse_data_strategy(pop_field(cfg, 'splits'), output_path)
    check_is_empty(cfg)
    return DataSetup(provider, strategy, data_filter)


def parse_data_strategy(cfg: ContextAwareDict, output_path: Path) -> DataStrategy:
    """
    Instantiate specific implementation of `DataStrategy` as configured.

    The `strategy` field is parsed, which is expected to refer to a name in
    the registry. See for example `lir.data_setup.binary_cross_validation`
    or `lir.data_setup.binary_train_test_split`.

    Data setup configuration is provided under the `data_setup` key.

    Parameters
    ----------
    cfg : ContextAwareDict
        Data strategy configuration.
    output_path : Path
        Output path for created objects.

    Returns
    -------
    DataStrategy
        Parsed data strategy instance.
    """
    strategy = pop_field(cfg, 'strategy')

    try:
        parser = registry.get(
            strategy,
            search_path=['data_strategies'],
            default_config_parser=GenericConfigParser,
        )
    except Exception as e:
        raise YamlParseError(
            cfg.context,
            f'no parser available for data strategy `{strategy}`; the error was: {e}',
        )

    return parser.parse(cfg, output_path)


class _FunctionDataProvider(DataProvider):
    def __init__(self, fn: Callable[[], InstanceData]):
        self.fn = fn

    def get_instances(self) -> InstanceData:
        return self.fn()


class GenericDataProviderConfigParser(GenericConfigParser):
    """
    Generic parser class to parse a data provider configuration into a  :class:`~lir.DataProvider` object.

    Parameters
    ----------
    component_class : type
        DataProvider class or callable to prepare data.
    """

    def __init__(self, component_class: type):
        super().__init__(component_class)

    def parse(
        self,
        config: ContextAwareDict,
        output_dir: Path,
    ) -> DataProvider:
        """
        Prepare a data provider function.

        Parameters
        ----------
        config : ContextAwareDict
            Constructor arguments for the component class.
        output_dir : Path
            Unused output directory argument required by parser API.

        Returns
        -------
        DataProvider
            Function that prepares and returns the data.
        """
        if issubclass(self.component_class, DataProvider):
            # if the component class is a subclass of DataProvider, instantiate it and return its `get_instances` method
            try:
                return self.component_class(**config)
            except Exception as e:
                raise YamlParseError(
                    config.context,
                    f'failed to instantiate module {self.component_class.__name__}: {e}',
                )

        elif callable(self.component_class):
            # if the component class is a callable, return it with the configuration attributes
            return _FunctionDataProvider(
                functools.partial(self.component_class, **config) if config else self.component_class
            )

        # don't know how to turn this into data
        raise YamlParseError(config.context, f'unrecognized data provider type: `{self.component_class}`')


def parse_data_provider(cfg: ContextAwareDict, output_path: Path) -> DataProvider:
    """
    Instantiate specific implementation of `DataProvider` as configured.

    The `method` field is parsed, which is expected to refer to a name in
    the registry. See for example `lir.config.data_sources.synthesized_normal_binary`
    or `lir.config.data_sources.synthesized_normal_multiclass`.

    Data sources are provided under the `data_sources` key.

    Parameters
    ----------
    cfg : ContextAwareDict
        Data provider configuration.
    output_path : Path
        Output path for created objects.

    Returns
    -------
    DataProvider
        Parsed data provider instance.
    """
    provider = pop_field(cfg, 'method')

    try:
        parser = registry.get(
            provider,
            search_path=['data_providers'],
            default_config_parser=GenericDataProviderConfigParser,
        )
    except Exception as e:
        raise YamlParseError(
            cfg.context,
            f'no parser available for data provider `{provider}`; the error was: {e}',
        )

    return parser.parse(cfg, output_path)
