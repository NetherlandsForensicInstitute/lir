import inspect
from pathlib import Path

from lir import registry
from lir.config.base import (
    ConfigParser,
    ContextAwareDict,
    YamlParseError,
    check_not_none,
    config_parser,
    pop_field,
)
from lir.transform import (
    BinaryClassifierTransformer,
    CsvWriter,
    FunctionTransformer,
    Identity,
    NumpyTransformer,
    Transformer,
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
                    f'failed to instantiate module {self.component_class.__name__}: {e}',
                )

            if isinstance(instance, Transformer):
                # The component already supports all necessary methods,
                # through the `Transformer` interface.
                return instance
            if hasattr(instance, 'transform'):
                # The component implements a `transform()` method, which means it
                # is a transformer and can be used in the scikit-learn pipeline.
                return instance
            if hasattr(instance, 'predict_proba'):
                # The component has a `predict_proba` method, which should be used as
                # `transform()` step in the pipeline, which the wrapper class provides.
                return BinaryClassifierTransformer(instance)

        elif callable(self.component_class):
            # When none of the above conditions apply, the component class might be a function
            # or a callable class, which should be used as a `transform()` step in the pipeline,
            # which the wrapper provides.
            return FunctionTransformer(self.component_class)

        raise YamlParseError(config.context, f'unrecognized module type: `{self.component_class}`')


class NumpyWrappingConfigParser(ConfigParser):
    """Wrap a Transformer to add a header to FeatureData."""

    def __init__(self, module_parser: ConfigParser):
        super().__init__()
        self.module_parser = module_parser

    def parse(self, config: ContextAwareDict, output_dir: Path) -> Transformer:
        """Parse the provided header configuration."""
        header = config.pop('header') if 'header' in config else None
        return NumpyTransformer(
            self.module_parser.parse(config, output_dir),
            header=header,
        )

    def reference(self) -> str:
        """Return the full name of the `module_parser` class argument."""
        return self.module_parser.reference()


def parse_module(
    module_config: ContextAwareDict | str | None,
    output_dir: Path,
    config_context_path: list[str],
    default_method: str | None = None,
) -> Transformer:
    """
    Construct a ``Transformer`` from a string or configuration section.

    If ``module_config`` is ``None``, an :class:`Identity` transformer is returned.

    If ``module_config`` is a dictionary, it must contain a ``method`` field whose
    value is the name of an object looked up in the registry. All remaining fields
    are passed as initialisation arguments. If no arguments are required, the input
    may be given directly as the object name.

    The resolved object is handled as follows:

    - If it is a subclass of :class:`ConfigParser`, the class is instantiated and the
      result of its :meth:`parse` method is returned.
    - If it defines a ``transform`` method, or is a subclass of ``Transformer``, it
      is instantiated and returned.
    - If it defines a ``predict_proba`` method, it is instantiated, wrapped in
      :class:`EstimatorTransformer`, and returned.
    - Any other callable is wrapped in :class:`FunctionTransformer` and returned.

    If ``module_config`` is a string, this function behaves as if a dictionary with a
    single field ``method`` set to that string had been provided.

    Parameters
    ----------
    module_config : dict or str or None
        Specification of the module.
    output_dir : str or pathlib.Path
        Directory where any output produced by the module is written.
    config_context_path : str
        Context path of this configuration, used for error reporting.
    default_method : str, optional
        Default value for the ``method`` field if it is not provided.

    Returns
    -------
    Transformer
        The constructed transformer instance.
    """
    if module_config is None:
        return Identity()
    elif isinstance(module_config, str):
        class_name = module_config
        args = ContextAwareDict(config_context_path)
    else:
        args = module_config
        class_name = pop_field(args, 'method', default=default_method, validate=check_not_none)

    return registry.get(class_name, GenericTransformerConfigParser, search_path=['modules']).parse(args, output_dir)


@config_parser
def csv_writer(config: ContextAwareDict, output_dir: Path) -> CsvWriter:
    """Set up a CSV Writer object, according to the configuration."""
    if 'path' not in config:
        config |= {'path': output_dir / f'{config.context[-1]}.csv'}
    return CsvWriter(**config)
