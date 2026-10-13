from pathlib import Path
from typing import Self

from lir import InstanceData, Transformer
from lir.config.base import ConfigValue, config_parser, pop_field
from lir.config.transform import parse_module


class Tee(Transformer):
    """
    Implementation of a custom transformer allowing to perform two separate tasks on a given input.

    Parameters
    ----------
    transformers : list[Transformer]
        Collection of transformers applied in sequence or parallel.
    """

    def __init__(self, transformers: list[Transformer]):
        super().__init__()
        self.transformers = transformers

    def fit(self, instances: InstanceData) -> Self:
        """
        Delegate `fit()` to all specified transformers.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        Self
            This tee transformer instance after delegating fit.
        """
        for transformer in self.transformers:
            transformer.fit(instances)

        return self

    def apply(self, instances: InstanceData) -> InstanceData:
        """
        Delegate `apply()` to all specified transformers.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        InstanceData
            Instance data object produced by this operation.
        """
        for transformer in self.transformers:
            transformer.apply(instances)

        return instances


@config_parser
def parse(
    config: ConfigValue,
    output_dir: Path,
) -> Tee:
    """
    Read configuration for modules section and provide wrapped corresponding transformers.

    Parameters
    ----------
    config : ConfigValue
        Configuration for the ``Tee`` transformer, containing a ``modules`` field with a list of module configurations.
    output_dir : Path
        Output directory for the parsed modules.

    Returns
    -------
    Tee
        A ``Tee`` transformer wrapping the parsed modules.
    """
    transformers = []
    modules = pop_field(config, 'modules')
    for module_config in modules:
        transformers.append(parse_module(module_config, output_dir))

    return Tee(transformers)
