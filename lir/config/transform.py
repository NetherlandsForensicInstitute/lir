import inspect
from pathlib import Path
from typing import Any, List, Mapping

from lir import registry
from lir.config.base import (
    config_parser,
    ConfigParser,
    YamlParseError,
    pop_field,
)
from lir.transform import (
    CsvWriter,
    NumpyTransformer,
    Transformer,
    FunctionTransformer,
    BinaryClassifierTransformer,
)


class GenericTransformerConfigParser(ConfigParser):
    """Parser class to help parse the defined component into its corresponding `Transformer` object.

    Since the scikit-learn `Pipeline` expects a `fit()` and `transform()` method on each of the pipeline steps,
    the configured components should adhere to this contract and implement these methods.

    The `parse()` function offered in this helper class, implements a branching strategy to determine
    which strategy is best suited to make the component compatible with the scikit-learn pipeline.
    """

    def __init__(self, component_class: object):
        super().__init__()
        self.component_class = component_class

    def parse(
        self,
        config: Mapping[str, Any],
        config_context_path: List[str],
        output_dir: Path,
    ) -> Transformer:
        """Prepare the defined component to support the expected methods in the scikit-learn `Pipeline`."""
        if inspect.isclass(self.component_class):
            try:
                instance = self.component_class(**config)
            except Exception as e:
                raise YamlParseError(
                    config_context_path,
                    f"failed to instantiate module {self.component_class.__name__}: {e}",
                )

            if isinstance(instance, Transformer):
                # The component already supports all necessary methods,
                # through the `Transformer` interface.
                return instance
            if hasattr(instance, "transform"):
                # The component implements a `transform()` method, which means it
                # is a transformer and can be used in the scikit-learn pipeline.
                return instance
            if hasattr(instance, "predict_proba"):
                # The component has a `predict_proba` method, which should be used as
                # `transform()` step in the pipeline, which the wrapper class provides.
                return BinaryClassifierTransformer(instance)

        elif callable(self.component_class):
            # When none of the above conditions apply, the component class might be a function
            # or a callable class, which should be used as a `transform()` step in the pipeline,
            # which the wrapper provides.
            return FunctionTransformer(self.component_class)

        raise YamlParseError(
            config_context_path, f"unrecognized module type: `{self.component_class}`"
        )


class NumpyCsvWriterWrappingConfigParser(ConfigParser):
    """Wrap a given CSV Writer object to handle Numpy specific data."""

    def __init__(self, module_parser: ConfigParser):
        super().__init__()
        self.module_parser = module_parser

    def parse(
        self, config: dict[str, Any], config_context_path: List[str], output_dir: Path
    ) -> Transformer:
        header = config.pop("header") if "header" in config else None
        path = (
            config.pop("path") if "path" in config else f"{config_context_path[-1]}.csv"
        )
        return NumpyTransformer(
            self.module_parser.parse(config, config_context_path, output_dir),
            header=header,
            path=output_dir / path,
        )


def parse_module(
    module_config: dict[str, Any] | str,
    config_context_path: List[str],
    output_dir: Path,
) -> Transformer:
    """
    Constructs a `Transformer` from a string or configuration section.

    The configuration section must have the field `method`, which is an object that name is looked up the registry. All
    other fields are initialization arguments. If no arguments are required, the input can be just the object name
    instead.

    If the class is:
    - a subclass of `ConfigParser, then the class is instantiated, and the return value of its `parse()` method is
      returned;
    - a class which has a `transform` attribute, or a `Transformer` subclass, it is instantiated and returned;
    - a class which has a `predict_proba` attribute, it is instantiated, wrapped by `EstimatorTransformer` and
      returned;
    - any other callable, it is wrapped by `FunctionTransformer`, and returned.
    """
    if isinstance(module_config, str):
        class_name = module_config
        args = {}
    else:
        args = dict(module_config)
        class_name = pop_field(config_context_path, args, "method")

    return registry.get(
        class_name, GenericTransformerConfigParser, search_path=["modules"]
    ).parse(args, config_context_path, output_dir)


@config_parser
def csv_writer(
    config: dict[str, Any], config_context_path: List[str], output_dir: Path
) -> CsvWriter:
    """Set up a CSV Writer object, according to the configuration."""
    if "path" not in config:
        config |= {"path": output_dir / f"{config_context_path[-1]}.csv"}
    return CsvWriter(**config)
