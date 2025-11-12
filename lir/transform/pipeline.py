from pathlib import Path
from typing import Any

from lir.config.base import ContextAwareDict, YamlParseError, check_is_empty, config_parser, pop_field
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

    def fit(self, instances: FeatureData) -> 'Pipeline':
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


@config_parser
def parse_pipeline(modules_config: ContextAwareDict, output_dir: Path) -> Pipeline:
    """Construct a scikit-learn Pipeline based on the provided configuration."""
    if modules_config is None:
        return Pipeline([])

    steps = pop_field(modules_config, 'steps')
    if not isinstance(steps, ContextAwareDict):
        raise YamlParseError(
            modules_config.context, f'invalid value for "steps": expected `dict`; found: {type(steps)}'
        )
    module_names = list(steps.keys())

    modules = [
        (
            module_name,
            parse_module(
                pop_field(steps, module_name),
                output_dir,
                steps.context + [module_name],
            ),
        )
        for module_name in module_names
    ]

    check_is_empty(modules_config)

    return Pipeline(modules)
