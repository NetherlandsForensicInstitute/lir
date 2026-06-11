import logging
from pathlib import Path

import numpy as np

from lir import InstanceData, Transformer
from lir.config.base import ContextAwareDict, config_parser, pop_field
from lir.data.io import DataFileBuilderCsv


LOG = logging.getLogger(__name__)


class CsvWriter(Transformer):
    """
    Implementation of a transformation step in a scikit-learn Pipeline that writes to CSV.

    This might be used to obtain temporary or intermediate results for logging or debugging
    purposes.

    Parameters
    ----------
    path : Path
        Filesystem path used by this operation.
    include_batch_number : bool
        The batch number is the sequence number of the call to `apply()`. Iff `include_batch`, this value is included as
        a column in the CSV file (default: True).
    include_fields : list[str] | None
        Fields to be included, or include all fields if not specified.
    exclude_fields : list[str] | None
        Optional list of fields to be excluded.
    """

    def __init__(
        self,
        path: Path | str,
        include_batch_number: bool = True,
        include_fields: list[str] | None = None,
        exclude_fields: list[str] | None = None,
    ):
        super().__init__()
        self.path = Path(path)
        if self.path is None:
            raise ValueError('missing argument: path')

        self.include_batch_number = include_batch_number
        self.include_fields = include_fields
        self.exclude_fields = exclude_fields
        self.n_batches = 0

    def fit_apply[DataType: InstanceData](self, instances: DataType) -> DataType:
        """
        Provide required `fit_apply()` and return all instances.

        Since the CsvWriter is implemented as a step (Transformer) in the pipeline, it should support
        the `fit_apply` method which is called on all transformers of the pipeline.

        This `fit_apply` method is typically called during training, and output is supposed to be generated for the test
        set only, so no output is generated at this stage. We also don't need to actually fit or transform anything, so
        we simply return the instances (as is).

        Parameters
        ----------
        instances : InstanceDataType
            Input instances to be processed by this method.

        Returns
        -------
        InstanceDataType
            Instance data object produced by this operation.
        """
        return instances

    def _add_fields(self, csv_builder: DataFileBuilderCsv, instances: InstanceData, fields: list[str]) -> None:
        for field in fields:
            if not self.exclude_fields or field not in self.exclude_fields:
                csv_builder.add_column(getattr(instances, field), field)

    def apply[DataType: InstanceData](self, instances: DataType) -> DataType:
        """
        Write numpy feature vector to CSV output file.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        InstanceData
            FeatureData object parsed from the source.
        """
        # initialize the csv builder
        write_mode = 'w' if self.n_batches == 0 else 'a'
        csv_builder = DataFileBuilderCsv(self.path, write_mode=write_mode)

        # add column: batch
        if self.include_batch_number:
            csv_builder.add_column(np.full((len(instances), 1), self.n_batches), 'batch')

        if self.include_fields is not None:
            # add listed columns, except excluded columns
            self._add_fields(csv_builder, instances, self.include_fields)
        else:
            # add all columns, except excluded columns
            include_fields = [field for field in instances.all_fields if getattr(instances, field) is not None]
            self._add_fields(csv_builder, instances, include_fields)

        # write all data that we accumulated
        csv_builder.write()

        self.n_batches += 1

        return instances


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
    path = output_dir / pop_field(config, 'path')
    return CsvWriter(path=path, **config)
