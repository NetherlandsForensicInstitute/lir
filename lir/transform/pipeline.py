from typing import Any

from lir.data.models import FeatureData
from lir.transform import as_transformer, Transformer


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
        for name, module in self.steps:
            instances = module.transform(instances)
        return instances

    def fit_transform(self, instances: FeatureData) -> FeatureData:
        for _name, module in self.steps:
            instances = module.fit_transform(instances)
        return instances
