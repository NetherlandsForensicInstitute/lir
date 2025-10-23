from collections.abc import Callable, Sequence
from pathlib import Path

from lir import registry
from lir.config.base import (
    YamlParseError,
    config_parser,
    pop_field,
    check_is_empty,
    GenericConfigParser,
)
from lir.config.data_providers import parse_data_provider
from lir.config.substitution import ContextAwareDict
from lir.data.data_strategies import (
    BinaryCrossValidation,
    MulticlassCrossValidation,
    BinaryTrainTestSplit,
    MulticlassTrainTestSplit,
)
from lir.data.models import DataStrategy


def _parse_train_test_split(
    config: ContextAwareDict,
    output_path: Path,
    constructor: Callable,
) -> DataStrategy:
    data_source = parse_data_provider(pop_field(config, "source"), output_path)
    test_size = pop_field(config, "test_size")
    seed = config.pop("seed", None)
    check_is_empty(config)
    return constructor(data_source, test_size, seed)


@config_parser
def binary_train_test_split(config: ContextAwareDict, output_path: Path) -> DataStrategy:
    """
    Parse settings for a train/test split strategy for binary data.
    """
    return _parse_train_test_split(config, output_path, BinaryTrainTestSplit)


@config_parser
def multiclass_train_test_split(config: ContextAwareDict, output_path: Path) -> DataStrategy:
    """
    Parse settings for a train/test split strategy for multiclass data.
    """
    return _parse_train_test_split(config, output_path, MulticlassTrainTestSplit)


def _parse_cross_validation(
    config: ContextAwareDict,
    output_path: Path,
    constructor: Callable,
    extra_args: Sequence[str],
) -> DataStrategy:
    folds = int(pop_field(config, "folds"))
    data_source = parse_data_provider(pop_field(config, "source"), output_path)
    check_is_empty(config, extra_args)
    return constructor(data_source, folds, **config)


@config_parser
def binary_cross_validation(config: ContextAwareDict, output_path: Path) -> DataStrategy:
    """Initialize K-fold cross validation strategy by parsing the configuration for binary classes.

    This method might be referenced in the YAML registry as follows:
    ```
    data_setup:
      binary_cross_validation: lir.config.data_setup.binary_cross_validation
    ```

    In the benchmark configuration YAML, this validation can be referenced as follows:
    ```
    binary_cross_validation_splits:
        setup: binary_cross_validation
        source: ${data}
    ```

    which is ultimately passed through the benchmark definition:
    ```
    benchmarks:
      model_selection_run:
        strategy: grid
        lr_system: ${lr_system}
        data: ${binary_cross_validation_splits}
        ...
    """
    return _parse_cross_validation(config, output_path, BinaryCrossValidation, ["seed"])


@config_parser
def multiclass_cross_validation(config: ContextAwareDict, output_path: Path) -> DataStrategy:
    """Initialize K-fold cross validation strategy by parsing the configuration for multiple classes.

    This method might be referenced in the YAML registry as follows:
    ```
    data_setup:
      multiclass_cross_validation: lir.config.data_setup.multiclass_cross_validation
    ```

    In the benchmark configuration YAML, this validation can be referenced as follows:
    ```
    cross_validation_splits:
        setup: multiclass_cross_validation
        folds: 5
        source: ${data}
    ```

    which is ultimately passed through the benchmark definition:
    ```
    benchmarks:
      model_selection_run:
        strategy: grid
        lr_system: ${lr_system}
        data: ${cross_validation_splits}
        ...
    ```
    """
    return _parse_cross_validation(config, output_path, MulticlassCrossValidation, [])


def parse_data_strategy(cfg: ContextAwareDict, output_path: Path) -> DataStrategy:
    """Instantiate specific implementation of `DataSetup` as configured.

    The `setup` field is parsed, which is expected to refer to a name in
    the registry. See for example `lir.data_setup.binary_cross_validation`
    or `lir.data_setup.binary_train_test_split`.

    Data setup configuration is provided under the `data_setup` key.
    """
    strategy = pop_field(cfg, "strategy")

    try:
        parser = registry.get(
            strategy,
            search_path=["data_strategies"],
            default_config_parser=GenericConfigParser,
        )
    except Exception as e:
        raise YamlParseError(
            cfg.context,
            f"no parser available for data type `{strategy}`; the error was: {e}",
        )

    return parser.parse(cfg, output_path)
