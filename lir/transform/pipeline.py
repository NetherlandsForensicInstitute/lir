from pathlib import Path
from typing import Any, Self

from lir.config.base import ContextAwareDict, YamlParseError, check_is_empty, check_not_none, config_parser, pop_field
from lir.config.transform import parse_module
from lir.data.models import FeatureData
from lir.transform import Transformer, as_transformer


class Pipeline(Transformer):
    """
    A pipeline of processing modules.

    A module may be a scikit-learn style transformer, estimator, or a LIR `Transformer`
    """

    def __init__(self, steps: list[tuple[str, Transformer | Any]]):
        """
        Constructor.

        :param steps: the steps of the pipeline as a list of (name, module) tuples.
        """
        self.steps = [(name, as_transformer(module)) for name, module in steps]

    def fit(self, instances: FeatureData) -> Self:
        for _name, module in self.steps[:-1]:
            instances = module.fit_transform(instances)

        if len(self.steps) > 0:
            _, last_module = self.steps[-1]
            last_module.fit(instances)

        return self

    def transform(self, instances: FeatureData) -> FeatureData:
        for _name, module in self.steps:
            instances = module.transform(instances)
        return instances

    def fit_transform(self, instances: FeatureData) -> FeatureData:
        for _name, module in self.steps:
            instances = module.fit_transform(instances)
        return instances


def parse_steps(config: ContextAwareDict, output_dir: Path) -> list[tuple[str, Transformer]]:
    if not isinstance(config, ContextAwareDict):
        raise YamlParseError(config.context, f'invalid value for "steps": expected `dict`; found: {type(config)}')
    module_names = list(config.keys())
    return [
        (
            module_name,
            parse_module(
                pop_field(config, module_name),
                output_dir,
                config.context + [module_name],
            ),
        )
        for module_name in module_names
    ]


@config_parser
def pipeline(config: ContextAwareDict, output_dir: Path) -> Pipeline:
    """Construct a scikit-learn Pipeline based on the provided configuration."""
    if config is None:
        return Pipeline([])

    steps = parse_steps(pop_field(config, 'steps', validate=check_not_none), output_dir)

    check_is_empty(config)
    return Pipeline(steps)
