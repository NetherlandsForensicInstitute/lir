import inspect
from pathlib import Path

from lir import registry
from lir.config.base import (
    ConfigParser,
    ContextAwareDict,
    GenericConfigParser,
    YamlParseError,
    check_not_none,
    config_parser,
    pop_field,
)
from lir.transform import (
    CsvWriter,
    FunctionTransformer,
    Identity,
    NumpyTransformer,
    Transformer,
    as_transformer,
)
from lir.transform.pairing import PairingMethod


class GenericTransformerConfigParser(ConfigParser):
    """
    Parser class to help parse the defined component into its corresponding `Transformer` object.

    Since the scikit-learn `Pipeline` expects a `fit()` and `transform()` method on each of the pipeline steps,
    the configured components should adhere to this contract and implement these methods.

    The `parse()` function offered in this helper class, implements a branching strategy to determine
    which strategy is best suited to make the component compatible with the scikit-learn pipeline.

    Parameters
    ----------
    component_class : object
        Component class or callable to adapt to the transformer interface.
    """

    def __init__(self, component_class: object):
        super().__init__()
        self.component_class = component_class

    def parse(
        self,
        config: ContextAwareDict,
        output_dir: Path,
    ) -> Transformer:
        """
        Prepare a configured component for use in a scikit-learn pipeline.

        Parameters
        ----------
        config : ContextAwareDict
            Constructor arguments for the component class.
        output_dir : Path
            Unused output directory argument required by parser API.

        Returns
        -------
        Transformer
            Component adapted to the ``Transformer`` interface.
        """
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

        elif callable(self.component_class):
            # When none of the above conditions apply, the component class might be a function
            # or a callable class, which should be used as a `transform()` step in the pipeline,
            # which the wrapper provides.
            return FunctionTransformer(self.component_class)

        raise YamlParseError(config.context, f'unrecognized module type: `{self.component_class}`')


class NumpyWrappingConfigParser(ConfigParser):
    """
    Wrap a Transformer to add a header to FeatureData.

    Parameters
    ----------
    module_parser : ConfigParser
        Parser used to create the wrapped transformer.
    """

    def __init__(self, module_parser: ConfigParser):
        super().__init__()
        self.module_parser = module_parser

    def parse(self, config: ContextAwareDict, output_dir: Path) -> Transformer:
        """
        Parse the provided header configuration.

        Parameters
        ----------
        config : ContextAwareDict
            Configuration possibly containing ``header`` and module fields.
        output_dir : Path
            Output directory passed to the wrapped parser.

        Returns
        -------
        Transformer
            Wrapped transformer that preserves numpy headers.
        """
        header = config.pop('header') if 'header' in config else None
        return NumpyTransformer(
            self.module_parser.parse(config, output_dir),
            header=header,
        )

    def reference(self) -> str:
        """
        Return the full name of the ``module_parser`` class argument.

        Returns
        -------
        str
            Reference string for the wrapped parser.
        """
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
    module_config : ContextAwareDict | str | None
        Specification of the module.
    output_dir : Path
        Directory where any output produced by the module is written.
    config_context_path : list[str]
        Context path of this configuration, used for error reporting.
    default_method : str | None, optional
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
    """
    Set up a CSV writer from configuration.

    Parameters
    ----------
    config : ContextAwareDict
        CSV writer configuration.
    output_dir : Path
        Output directory used to derive default CSV path.

    Returns
    -------
    CsvWriter
        Configured CSV writer.
    """
    if 'path' not in config:
        config |= {'path': output_dir / f'{config.context[-1]}.csv'}
    return CsvWriter(**config)


def parse_pairing_config(
    module_config: ContextAwareDict | str,
    output_dir: Path,
    context: list[str],
) -> PairingMethod:
    """
    Parse and delegate pairing to the corresponding function for the defined pairing method.

    The argument `module_config` defines the pairing method. If its value is a `str`, the registry is queried and the
    corresponding pairing method is returned. If its value is a `dict`, the pairing method is defined
    by the value `module_config["method"]`, and the registry is queried for the config parser of
    the corresponding pairing method. The remaining values in `module_config` are passed as arguments to the
    configuration parser of the pairing method.

    If the registry cannot resolve the pairing method, an exception is raised.

    Parameters
    ----------
    module_config : ContextAwareDict | str
        Pairing method configuration.
    output_dir : Path
        Output directory for parser calls.
    context : list[str]
        Context used when ``module_config`` is a string.

    Returns
    -------
    PairingMethod
        Parsed pairing method.
    """
    if isinstance(module_config, str):
        class_name = module_config
        args = ContextAwareDict(context)
    else:
        class_name = pop_field(module_config, 'method')
        args = module_config

    return registry.get(class_name, search_path=['pairing'], default_config_parser=GenericConfigParser).parse(
        args, output_dir
    )
