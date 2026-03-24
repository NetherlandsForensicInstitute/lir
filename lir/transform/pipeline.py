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


_LOG = logging.getLogger(__name__)


__all__ = [
    'Pipeline',
    'LoggingPipeline',
    'logging_pipeline',
]


class Pipeline(Transformer):
    """
    A pipeline of processing modules.

    Each step in the pipeline may be a

    - a scikit-learn style transformer (with ``fit()`` and ``transform()`` functions),
    - a scikit-learn style estimator (with ``fit()`` and ``predict_proba()``), or
    - a LiR ``Transformer`` object.

    Example:

    .. code-block:: python

        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from lir.transform.pipeline import Pipeline
        from lir.algorithms.logistic_regression import LogitCalibrator
        from lir.util import probability_to_odds

        pipeline = Pipeline(steps=[
                ('scaler', StandardScaler()),       # a scikit-learn transformer, for scaling the data
                ('clf', RandomForestClassifier()),  # a scikit-learn estimator, to calculate pseudo-probabilities
                ('to_odds', probability_to_odds),   # a plain function, to convert probabilities to pseudo-LLRs
                ('calibrator', LogitCalibrator()),  # a LiR transformer, to calibrate the LLRs
        ])

    Parameters
    ----------
    steps : list[tuple[str, Transformer | Any]]
        Ordered transformer steps executed by this pipeline.
    """

    def __init__(self, steps: list[tuple[str, Transformer | Any]]):
        self.steps = [(name, as_transformer(module)) for name, module in steps]

    def fit(self, instances: InstanceData) -> Self:
        """
        Fit the model on the instance data.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        Self
            The fitted pipeline instance.
        """
        for _name, module in self.steps[:-1]:
            instances = module.fit_apply(instances)

        if len(self.steps) > 0:
            _, last_module = self.steps[-1]
            last_module.fit(instances)

        return self

    def apply(self, instances: InstanceData) -> InstanceData:
        """
        Apply the fitted model on the instance data.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        InstanceData
            Instance data object produced by this operation.
        """
        for _name, module in self.steps:
            instances = module.apply(instances)
        return instances

    def fit_apply(self, instances: InstanceData) -> InstanceData:
        """
        Combine fitting the transformer/estimator and applying the model to the instances.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        InstanceData
            Instance data object produced by this operation.
        """
        for _name, module in self.steps:
            instances = module.fit_apply(instances)
        return instances


def parse_steps(config: ContextAwareDict | None, output_dir: Path) -> list[tuple[str, Transformer]]:
    """
    Parse the defined pipeline steps in the configuration and return the initialized modules as a list.

    Parameters
    ----------
    config : ContextAwareDict | None
        Configuration mapping used to construct this component.
    output_dir : Path
        Directory where generated outputs are written.

    Returns
    -------
    list[tuple[str, Transformer]]
        List of (name, module) tuples for the pipeline steps.
    """
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
    """
    Construct a scikit-learn Pipeline based on the provided configuration.

    Parameters
    ----------
    config : ContextAwareDict
        Configuration mapping used to construct this component.
    output_dir : Path
        Directory where generated outputs are written.

    Returns
    -------
    Pipeline
        The constructed pipeline instance.
    """
    if config is None:
        return Pipeline([])

    steps = parse_steps(pop_field(config, 'steps'), output_dir)

    check_is_empty(config)
    return Pipeline(steps)


class LoggingPipeline(Pipeline):
    """
    A pipeline that writes debugging output to a CSV file.

    This pipeline act like a normal ``Pipeline``, but has a CSV file as a side effect. Depending on the settings and the
    data, the CSV file may have the following columns:

    - ``batch``: if the data strategy yields multiple train/test splits, the batch value is the sequence number of the
      test set
    - ``label``: the hypothesis label

    In addition, there may be columns for the input features and output of individual steps. These columns are named
    ``featuresI`` for input features or ``stepnameI`` for step output, where ``stepname`` is replaced by the name of the
    step, and ``I`` refers to the index of the feature value.

    Parameters
    ----------
    steps : list[tuple[str, Transformer | Any]]
        Ordered transformer steps executed by this pipeline.
    output_file : PathLike
        Destination file used to log intermediate pipeline output.
    include_batch_number : bool
        Whether to include the batch number in logged output.
    include_labels : bool
        Whether to include labels in logged output.
    include_fields : list[str] | None
        Additional instance fields to include in logged output.
    include_steps : list[str] | None
        Whether to include step names in logged output.
    include_input : bool
        Whether to include original inputs in logged output.
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
        super().__init__(steps)
        self.output_file = Path(output_file)
        self.include_batch_number = include_batch_number
        self.include_labels = include_labels
        self.include_fields = include_fields or []
        self.include_steps = set(include_steps or list(zip(*steps, strict=True))[0])
        self.include_input = include_input
        self.n_batches = 0

    def apply(self, instances: InstanceData) -> InstanceData:
        """
        Apply the pipeline to the incoming instances.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        InstanceData
            Instance data object produced by this operation.
        """
        # initialize the csv builder
        write_mode = 'w' if self.n_batches == 0 else 'a'
        csv_builder = DataFileBuilderCsv(self.output_file, write_mode=write_mode)

        # add column: batch
        if self.include_batch_number:
            csv_builder.add_column(np.full((len(instances), 1), self.n_batches), 'batch')

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


@config_parser(reference=LoggingPipeline)
def logging_pipeline(config: ContextAwareDict, output_dir: Path) -> Pipeline:
    """
    Construct a scikit-learn Pipeline based on the provided configuration.

    Parameters
    ----------
    config : ContextAwareDict
        Configuration mapping used to construct this component.
    output_dir : Path
        Directory where generated outputs are written.

    Returns
    -------
    Pipeline
        Logging pipeline configured from the provided YAML section.
    """
    if config is None:
        return Pipeline([])

    steps = parse_steps(pop_field(config, 'steps'), output_dir)
    output_file = output_dir / pop_field(config, 'output_file', default=f'{config.context[-1]}.csv')

    return LoggingPipeline(steps, output_file, **config)
