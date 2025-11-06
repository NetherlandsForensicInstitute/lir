import csv
import logging
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import Any, Self

import numpy as np

from lir.config.base import ContextAwareDict, YamlParseError, check_is_empty, config_parser, pop_field
from lir.config.transform import parse_module
from lir.data.models import FeatureData
from lir.transform import Transformer, as_transformer


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

    steps = parse_steps(pop_field(config, 'steps'), output_dir)

    check_is_empty(config)
    return Pipeline(steps)


class DebugPipeline(Pipeline):
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

    @staticmethod
    def _append_data(
        all_headers: list[str], all_data: list[np.ndarray], name: str, data: np.ndarray, header: list[str] | None = None
    ) -> None:
        """
        Append data and corresponding headers to `all_data` and `all_headers`.
        """
        if len(data.shape) != 2:
            data = data.reshape(data.shape[0], -1)

        if isinstance(header, list) and len(header) == data.shape[1]:
            all_headers.extend(header)
        elif data.shape[1] == 1:
            all_headers.append(name)
        else:
            all_headers.extend([f'{name}{i}' for i in range(data.shape[1])])

        all_data.append(data)

    def transform(self, instances: FeatureData) -> FeatureData:
        LOG.info(f'writing CSV file: {self.output_file}')
        self.output_file.parent.mkdir(exist_ok=True, parents=True)
        if self.n_batches == 0:
            write_mode = 'w'
            write_header = True
        else:
            write_mode = 'a'
            write_header = False

        all_headers: list[str] = []
        all_data: list[np.ndarray] = []

        if self.include_batch_number:
            self._append_data(all_headers, all_data, 'batch', np.ones((len(instances), 1)) * self.n_batches)

        if self.include_labels and instances.labels is not None:
            self._append_data(all_headers, all_data, 'label', instances.labels)

        if self.include_input:
            self._append_data(all_headers, all_data, 'features', instances.features)

        for module_name, module in self.steps:
            instances = module.transform(instances)

            if module_name in self.include_steps:
                self._append_data(
                    all_headers, all_data, module_name, instances.features, getattr(instances, 'header', None)
                )

        for field in self.include_fields:
            values = getattr(instances, field)
            if not isinstance(values, np.ndarray):
                raise ValueError(f'expected type: np.ndarray; found: {type(values)}')
            self._append_data(all_headers, all_data, field, values)

        with open(self.output_file, write_mode) as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow(all_headers)

            for row in zip(*all_data, strict=True):
                writer.writerow(chain(*row))

        self.n_batches += 1

        return instances


@config_parser
def debug_pipeline(config: ContextAwareDict, output_dir: Path) -> Pipeline:
    """Construct a scikit-learn Pipeline based on the provided configuration."""
    if config is None:
        return Pipeline([])

    steps = parse_steps(pop_field(config, 'steps'), output_dir)
    output_file = output_dir / pop_field(config, 'output_file', default=f'{config.context[-1]}.csv')

    return DebugPipeline(steps, output_file, **config)
