from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from lir import registry
from lir.config.base import (
    ConfigParser,
    ConfigValue,
    GenericConfigParser,
    YamlParseError,
    get_full_name,
)
from lir.config.util import parse_config
from lir.data.models import DataProvider, DataSetup, DataStrategy, InstanceData


def parse_data_setup(cfg: ConfigValue, output_path: Path) -> DataSetup:
    """
    Parse data provider and data strategy from configuration.

    The fields `provider`, `filter` and `splits` are parsed, which are expected to refer
    to specific implementations of `DataProvider`, `Transformer` and `DataStrategy`, respectively.
    See `parse_data_provider`, `parse_module` and `parse_data_strategy` for more information.

    Parameters
    ----------
    cfg : ConfigValue
        Configuration section containing provider and split strategy.
    output_path : Path
        Output path for created objects.

    Returns
    -------
    DataSetup
        Parsed data provider, filter and strategy.
    """
    return parse_config(
        cfg,
        output_path,
        method_key='setup',
        default_method='split_data',
        default_config_parser=GenericConfigParser,
        search_path=['data_setup'],
    )


def parse_splitting_strategy(cfg: ConfigValue, output_path: Path) -> DataStrategy:
    """
    Instantiate specific implementation of `DataStrategy` as configured.

    The `strategy` field is parsed, which is expected to refer to a name in
    the registry. See for example :class:`lir.data_strategies.CrossValidation`
    or :class:`lir.data_strategies.TrainTestSplit`.

    Data strategy configuration is provided under the `data.splits` key.

    Parameters
    ----------
    cfg : ConfigValue
        Data strategy configuration.
    output_path : Path
        Output path for created objects.

    Returns
    -------
    DataStrategy
        Parsed data strategy instance.
    """
    return parse_config(
        cfg,
        output_path,
        method_key='strategy',
        search_path=['data_strategies'],
        default_config_parser=GenericConfigParser,
    )


class _DataProviderFunction(DataProvider):
    def __init__(self, fn: Callable[[], InstanceData]):
        self.fn = fn

    def get_instances(self) -> InstanceData:
        return self.fn()


def data_provider[ReturnType: InstanceData](func: Callable[[ConfigValue, Path], ReturnType]) -> Callable:
    """
    Wrap a parsing function in a ``ConfigParser`` object using a decorator.

    The :meth:`parse` method of the resulting :class:`~lir.config.base.ConfigParser` instance returns a
    :class:`~lir.DataProvider` object that is invoked when needed.

    This decorator can be used as follows:

    .. code-block:: python

        @data_provider
        def foo(path: str, some_argument: int) -> InstanceData:
            with open(path) as f:
                ...
                return FeatureData(...)

    Parameters
    ----------
    func : Callable[[ConfigValue, Path], ReturnType]
        Function to wrap as a config parser.

    Returns
    -------
    Callable
        Decorator result or wrapped ``ConfigParser`` implementation.
    """

    class ConfigParserFunction(ConfigParser):
        __doc__ = func.__doc__

        def parse(
            self,
            config: ConfigValue,
            output_dir: Path,
        ) -> DataProvider:
            return _DataProviderFunction(partial(func, **config.as_dict()))

        def reference(self) -> str:
            return get_full_name(func)

        def __call__(self, *args: Any, **kwargs: Any) -> ReturnType:  # numpydoc ignore=RT01,PR01
            """Call the decorated function directly, as if it was not decorated."""
            return func(*args, **kwargs)

    return ConfigParserFunction()


def parse_data_provider(cfg: ConfigValue, output_path: Path) -> DataProvider:
    """
    Instantiate specific implementation of `DataProvider` as configured.

    The `method` field is parsed, which is expected to refer to a name in
    the registry. See for example `lir.datasets.synthesized_normal_binary`
    or `lir.datasets.synthesized_normal_multiclass`.

    Data sources are provided under the `data_providers` key.

    Parameters
    ----------
    cfg : ConfigValue
        Data provider configuration.
    output_path : Path
        Output path for created objects.

    Returns
    -------
    DataProvider
        Parsed data provider instance.
    """
    return parse_config(
        cfg,
        output_path,
        method_key='method',
        default_config_parser=GenericConfigParser,
        search_path=['data_providers'],
    )
