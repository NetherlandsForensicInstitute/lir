import logging
from os import PathLike
from pathlib import Path
from typing import Any, Self

import numpy as np

from lir.config.base import ContextAwareDict, YamlParseError, check_is_empty, config_parser, pop_field
from lir.config.transform import parse_module
from lir.data.io import DataFileBuilderCsv
from lir.data.models import FeatureData, InstanceData
from lir.transform import Transformer, as_transformer
from lir.util import check_type


LOG = logging.getLogger(__name__)


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

    def fit(self, instances: InstanceData) -> Self:
        for _name, module in self.steps[:-1]:
            instances = module.fit_apply(instances)

        if len(self.steps) > 0:
            _, last_module = self.steps[-1]
            last_module.fit(instances)

        return self

    def apply(self, instances: InstanceData) -> InstanceData:
        for _name, module in self.steps:
            instances = module.apply(instances)
        return instances

    def fit_apply(self, instances: InstanceData) -> InstanceData:
        for _name, module in self.steps:
            instances = module.fit_apply(instances)
        return instances


def parse_steps(config: ContextAwareDict | None, output_dir: Path) -> list[tuple[str, Transformer]]:
    if config is None:
        return []
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

    steps = parse_steps(pop_field(config, 'steps'), output_dir)

    check_is_empty(config)
    return Pipeline(steps)


class LoggingPipeline(Pipeline):
    """
    A pipeline that writes debugging output to a CSV file.
    """

    def __init__(
        self,
        steps: list[tuple[str, Transformer | Any]],
        output_file: PathLike,
        include_batch_number: bool = True,
        include_labels: bool = True,
        include_fields: list[str] | None = None,
        include_steps: list[str] | None = None,
        include_input: bool = True,
    ):
        """
        :param steps: the steps of the pipeline as a list of (name, module) tuples.
        :param output_file: the name of the generated output file
        :param include_batch_number: (bool) whether the zero-indexed batch number should be included in the output,
            e.g. for cross-validation (defaults to True)
        :param include_labels: (bool) whether the labels should be included, if available (defaults to True)
        :param include_fields: a list of the names of extra fields to include (defaults to none)
        :param include_steps: a list of steps to include (defaults to all)
        :param include_input: (bool) whether to write the input features (defaults to True)
        """
        super().__init__(steps)
        self.output_file = Path(output_file)
        self.include_batch_number = include_batch_number
        self.include_labels = include_labels
        self.include_fields = include_fields or []
        self.include_steps = set(include_steps or list(zip(*steps, strict=True))[0])
        self.include_input = include_input
        self.n_batches = 0

    def apply(self, instances: InstanceData) -> InstanceData:
        # initialize the csv builder
        write_mode = 'w' if self.n_batches == 0 else 'a'
        csv_builder = DataFileBuilderCsv(self.output_file, write_mode=write_mode)

        # add column: batch
        if self.include_batch_number:
            csv_builder.add_column(np.ones((len(instances), 1)) * self.n_batches, 'batch')

        # add column: label
        if self.include_labels and instances.labels is not None:
            csv_builder.add_column(instances.labels, 'label')

        # add columns: features
        if self.include_input:
            instances = check_type(FeatureData, instances, message='expected FeatureData as pipeline input')
            csv_builder.add_column(instances.features, 'features')

        try:
            # add columns for the output of each step
            for module_name, module in self.steps:
                instances = module.apply(instances)

                if module_name in self.include_steps:
                    instances = check_type(
                        FeatureData, instances, message=f'expected FeatureData as output of pipeline step {module_name}'
                    )
                    header = getattr(instances, 'header', None) or module_name
                    csv_builder.add_column(instances.features, header)

            # add columns for extra fields
            for field in self.include_fields:
                values = getattr(instances, field)
                if not isinstance(values, np.ndarray):
                    raise ValueError(f'expected type: np.ndarray; found: {type(values)}')
                csv_builder.add_column(values, field)

        finally:
            # write all data that we accumulated, even if some steps failed
            csv_builder.write()

        self.n_batches += 1
        return instances


@config_parser
def logging_pipeline(config: ContextAwareDict, output_dir: Path) -> Pipeline:
    """Construct a scikit-learn Pipeline based on the provided configuration."""
    if config is None:
        return Pipeline([])

    steps = parse_steps(pop_field(config, 'steps'), output_dir)
    output_file = output_dir / pop_field(config, 'output_file', default=f'{config.context[-1]}.csv')

    return LoggingPipeline(steps, output_file, **config)
