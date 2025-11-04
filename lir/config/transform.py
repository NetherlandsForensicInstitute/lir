import inspect
from pathlib import Path

from lir import registry
from lir.config.base import (
    config_parser,
    ConfigParser,
    YamlParseError,
    pop_field,
    ContextAwareDict,
)
from lir.transform import (
    CsvWriter,
    NumpyTransformer,
    Transformer,
    FunctionTransformer,
    as_transformer,
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
        config: ContextAwareDict,
        output_dir: Path,
    ) -> Transformer:
        """Prepare the defined component to support the expected methods in the scikit-learn `Pipeline`."""
        if inspect.isclass(self.component_class):
            try:
                instance = self.component_class(**config)

                # make sure we have an object of type `Transformer`
                return as_transformer(instance)
            except Exception as e:
                raise YamlParseError(
                    config.context,
                    f"failed to instantiate module {self.component_class.__name__}: {e}",
                )

        elif callable(self.component_class):
            # When none of the above conditions apply, the component class might be a function
            # or a callable class, which should be used as a `transform()` step in the pipeline,
            # which the wrapper provides.
            return FunctionTransformer(self.component_class)

        raise YamlParseError(config.context, f"unrecognized module type: `{self.component_class}`")


class NumpyCsvWriterWrappingConfigParser(ConfigParser):
    """Wrap a given CSV Writer object to handle Numpy specific data."""

    def __init__(self, module_parser: ConfigParser):
        super().__init__()
        self.module_parser = module_parser

    def parse(self, config: ContextAwareDict, output_dir: Path) -> Transformer:
        header = config.pop("header") if "header" in config else None
        path = pop_field(config, "path", default=f"{config.context[-1]}.csv")
        return NumpyTransformer(
            self.module_parser.parse(config, output_dir),
            header=header,
            path=output_dir / path,
        )


def parse_module(
    module_config: ContextAwareDict | str,
    output_dir: Path,
    config_context_path: list[str],
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
        args = ContextAwareDict(config_context_path)
    else:
        args = module_config
        class_name = pop_field(args, "method")

    return registry.get(class_name, GenericTransformerConfigParser, search_path=["modules"]).parse(args, output_dir)


@config_parser
def csv_writer(config: ContextAwareDict, output_dir: Path) -> CsvWriter:
    """Set up a CSV Writer object, according to the configuration."""
    if "path" not in config:
        config |= {"path": output_dir / f"{config.context[-1]}.csv"}
    return CsvWriter(**config)
