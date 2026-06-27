import csv
import io
import itertools
import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import IO, Any

import numpy as np
import requests
from requests_cache import CachedSession

from lir.config.base import ContextAwareDict, check_is_empty, config_parser, pop_field
from lir.data.io import search_path
from lir.data.models import DataProvider, FeatureData
from lir.data_strategies import RoleAssignment
from lir.util import check_is_enum_option, check_type


LOG = logging.getLogger(__name__)


class ParseError(ValueError):
    """
    Exception to be raised on parse errors.

    This happens when an input file is malformatted or contains invalid input.
    """


@dataclass
class DataField:
    """
    A data field for parsing a CSV file into an :class:`~lir.InstanceData` object.

    Attributes
    ----------
    field_name : str
        The attribute name in the ``InstanceData`` object.
    column_names : list[str]
        The associated column names in the CSV file.
    validate_cell : Callable[[str], Any]
        A validation function for parsing column values.
    """

    field_name: str
    column_names: list[str]
    validate_cell: Callable[[str], Any] = str

    def _get_validated_value(self, row: dict[str, str], column_name: str) -> Any:
        value = row.pop(column_name)
        try:
            return self.validate_cell(value)
        except (ValueError, TypeError) as e:
            raise ParseError(f'failed to parse {value} for column {column_name}: {e}') from e

    def parse_from_row(self, row: dict[str, str]) -> Any:
        """
        Parse a row into a field value.

        Parameters
        ----------
        row : dict[str, str]
            A row from the CSV file.

        Returns
        -------
        Any
            The parsed value, either a single value (int, float, str), or a list of values.
        """
        if len(self.column_names) == 1:
            return self._get_validated_value(row, self.column_names[0])
        else:
            return [self._get_validated_value(row, column_name) for column_name in self.column_names]


class ImplicitFeaturesField(DataField):
    """
    A features field for parsing a CSV file into an :class:`~lir.InstanceData` object.

    This field does not require feature columns to be specified explicitly, but assumes all columns not otherwise
    assigned to hold feature values.
    """

    def __init__(self) -> None:
        super().__init__('features', [], float)

    def parse_from_row(self, row: dict[str, str]) -> list[float]:
        """
        Parse a row into feature values.

        Parameters
        ----------
        row : dict[str, str]
            A row from the CSV file.

        Returns
        -------
        list[float]
            The feature values.
        """
        if not self.column_names:
            self.column_names = list(row.keys())

        return super().parse_from_row(row)


class ExtraField(DataField):  # numpydoc ignore=PR01
    """Extra field for CSV parsing."""

    def __init__(self, field_name: str, column_name: str, validate_cell: Callable[[str], Any]) -> None:
        super().__init__(field_name, [column_name], validate_cell)

    @property
    def column_name(self) -> str:  # numpydoc ignore=RT01
        """Return column name."""
        return self.column_names[0]


class FeatureDataCsvParser(DataProvider):
    """
    Parse a CSV file into an :class:`~lir.InstanceData` object.

    The CSV file contents are provided by the ``open_file_fn`` argument.

    Parameters
    ----------
    open_file_fn : Callable[[], IO]
        Function that returns a data stream from which the CSV file contents can be read.
    source_id_column : str | list[str] | None
        Column name(s) containing source identifiers (each source has a unique string identifier).
    label_column : str | None
        Column name containing hypothesis labels (value 0 for H2 or 1 for H2).
    feature_columns : str | list[str] | None
        Column names containing numerical feature values. If not specified, all columns not otherwise designated are
        interpreted as feature columns.
    instance_id_column : str | None
        Column name containing instance identifiers.
    role_assignment_column : str | None
        Column name containing predefined roles (value 'train' for training or 'test' for test).
    fold_assignment_column : str | None
        Column name containing predefined fold assignments (each fold has a unique string identifier).
    extra_fields : list[ExtraField] | None
        Optional extra fields to parse from each row.
    ignore_columns : list[str] | None
        Column names ignored when extracting features. This attribute is ignored if `feature_columns` is available.
    head : int | None
        Maximum number of rows to read from the source.
    message_prefix : str
        Prefix added to parser log and error messages.
    continue_on_error : bool
        If True, a row will be dropped if a parse error occurs. Otherwise, parsing will be aborted.

    Examples
    --------
    Assume a CSV file containing two features and source identifiers:

    .. code-block:: text

        source_id,feature1,feature2,feature3,name_of_an_irrelevant_column
        0,1,10,1,sherlock
        0,1,11,1,holmes
        1,20,30,1,irene
        1,18,32,3,adler
        2,5,10,8,professor
        2,1,11,8,moriarty

    This file can be parsed using the following YAML configuration:

    .. code-block:: yaml

        data:
          provider: parse_features_from_csv_file
            path: path/to/file.csv
            source_id_column: source_id
            ignore_columns:
              - name_of_an_irrelevant_column

    .. code-block:: yaml

        data:
          provider: parse_features_from_csv_url
            url: https://raw.githubusercontent.com/NetherlandsForensicInstitute/elemental_composition_glass/refs/heads/main/training.csv
            source_id_column: Item
            ignore_columns:
              - id
              - Piece
    """

    def __init__(
        self,
        open_file_fn: Callable[[], IO],
        source_id_column: str | list[str] | None = None,
        label_column: str | None = None,
        feature_columns: str | list[str] | None = None,
        instance_id_column: str | None = None,
        role_assignment_column: str | None = None,
        fold_assignment_column: str | None = None,
        extra_fields: list[ExtraField] | None = None,
        ignore_columns: list[str] | None = None,
        head: int | None = None,
        message_prefix: str = '',
        continue_on_error: bool = False,
    ):
        self._open_file_fn = open_file_fn
        self.data_fields: list[DataField] = []

        if isinstance(source_id_column, str):
            self.data_fields.append(DataField('source_ids', [source_id_column]))
        elif isinstance(source_id_column, list):
            if len(source_id_column) > 2:
                raise ValueError(f'the number of source id columns can be at most 2; found: {len(source_id_column)}')
            self.data_fields.append(DataField('source_ids', source_id_column))
        elif source_id_column is not None:
            raise ValueError(f'the source id column should be a string or a list; found: {type(source_id_column)}')

        if label_column:
            self.data_fields.append(DataField('labels', [label_column], int))
        if instance_id_column:
            self.data_fields.append(DataField('instance_ids', [instance_id_column]))
        if role_assignment_column:
            self.data_fields.append(
                DataField('role_assignments', [role_assignment_column], partial(check_is_enum_option, RoleAssignment))
            )
        if fold_assignment_column:
            self.data_fields.append(DataField('fold_assignments', [fold_assignment_column]))
        self.data_fields.extend(extra_fields or [])
        self.ignore_columns: list[str] = ignore_columns or []
        self._head = head
        self._message_prefix = message_prefix
        self.continue_on_error = continue_on_error

        if isinstance(feature_columns, list):
            self.data_fields.append(DataField('features', feature_columns, float))
        else:
            self.data_fields.append(ImplicitFeaturesField())

    def _parse_row(
        self, row: dict[str, str], reader: csv.DictReader
    ) -> dict[str, str | list[str] | int | float | list[float]]:
        fields: dict[str, str | list[str] | int | float | list[float]] = {}

        for column_name in self.ignore_columns:
            row.pop(column_name)
        for field in self.data_fields:
            fields[field.field_name] = field.parse_from_row(row)
        return fields

    def _parse_file(self, fp: IO) -> FeatureData:
        # initialize the reader
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ParseError(f'{self._message_prefix}empty file')

        # identify the non feature columns
        columns_with_explicit_role = [
            (field.field_name, column) for field in self.data_fields for column in field.column_names
        ] + [('ignore_columns', column_name) for column_name in self.ignore_columns]

        # check if all required columns exist in the csv file
        for name, value in columns_with_explicit_role:
            if value is not None and value not in reader.fieldnames:
                raise ParseError(
                    f'{self._message_prefix}{name} specified as `{value}`, but it is not present in the csv file'
                )

        # initialize the result values
        all_instances: dict[str, list[Any]] = {}

        # read the file, row by row
        n_instances = 0
        for row in itertools.islice(reader, self._head):
            try:
                fields = self._parse_row(row, reader)
                if not all_instances:
                    for key in fields:
                        all_instances[key] = []
                for k, v in fields.items():
                    all_instances[k].append(v)

                n_instances += 1
            except ParseError as e:
                if self.continue_on_error:
                    LOG.info(f'{self._message_prefix}line {reader.line_num}: parsing failed: {e}')
                else:
                    raise ParseError(f'{self._message_prefix}line {reader.line_num}: parsing failed: {e}')

        if not all_instances:
            return FeatureData(features=np.zeros((0, 1)))

        if self._head is not None and n_instances < self._head:
            LOG.warning(f'input file has too few rows; expected: {self._head}; found: {n_instances}')

        # finalize the data
        all_instances = {k: np.array(v) for k, v in all_instances.items()}
        return FeatureData(**all_instances)  # type: ignore

    def get_instances(self) -> FeatureData:
        """
        Retrieve FeatureData instances.

        Returns
        -------
        FeatureData
            FeatureData object parsed from the source.
        """
        LOG.debug(f'{self._message_prefix}parsing CSV file')
        with self._open_file_fn() as f:
            return self._parse_file(f)


def _parse_cell_type(value: str) -> Callable[[str], Any]:
    try:
        return {
            'int': int,
            'float': float,
            'str': str,
        }[value]
    except KeyError:
        raise ValueError(f'unknown cell type: {value}')


def _parse_extra_field(config: ContextAwareDict | str) -> ExtraField:
    """
    Parse an extra field configuration into an ExtraField object.

    The configuration can be:
     - a string: the column name and field name are assumed to be the same, and the cell type defaults to str.
     - a dictionary: it must contain the key 'column' and may optionally contain 'name' and 'cell_type'.

    Parameters
    ----------
    config : ContextAwareDict | str
        Configuration for the extra field.

    Returns
    -------
    ExtraField
        Parsed ExtraField object.
    """
    if isinstance(config, str):
        # If config is a string, we assume that the column name and the field name are the same (namely 'config').
        return ExtraField(config, config, str)

    if isinstance(config, ContextAwareDict):
        # If config is a dictionary, we expect it to contain the keys 'column'.
        column_name = pop_field(config, 'column', required=True)

        # The field_name is optional; if not provided, we use the column name as the field name.
        field_name = pop_field(config, 'name', default=column_name)

        # The cell_type is also optional; if not provided, we default to str.
        cell_type = pop_field(config, 'cell_type', validate=_parse_cell_type, default=str)

        check_is_empty(config)

        return ExtraField(field_name, column_name, cell_type)

    raise ValueError(f'Extra field expected str or dict, but got {type(config)} ')


def _parse_feature_data_csv(config: ContextAwareDict, **kwargs: Any) -> FeatureDataCsvParser:
    extra_fields_config = pop_field(config, 'extra_fields', default=[], validate=partial(check_type, list))
    extra_fields = [_parse_extra_field(field_config) for field_config in extra_fields_config]

    parser = FeatureDataCsvParser(**config, extra_fields=extra_fields, **kwargs)
    return parser


@config_parser
def feature_data_csv_http_parser(config: ContextAwareDict, output_dir: Path) -> FeatureDataCsvParser:
    """
    Initialize the CSV parser that reads data from a stream.

    Arguments:
    - use_cache: boolean indicating whether to cache retrieved data

    Other arguments are passed directly to `FeatureDataCsvParser`.

    Parameters
    ----------
    config : ContextAwareDict
        Configuration mapping used to construct this component.
    output_dir : Path
        Directory where generated outputs are written.

    Returns
    -------
    FeatureDataCsvHttpParser
        FeatureData object parsed from the source.
    """
    use_cache = pop_field(config, 'use_cache', default=True, validate=partial(check_type, bool))

    session: requests.Session
    if use_cache:
        session = CachedSession('lir', use_cache_dir=True)
        LOG.debug(f'using cache location: {session.cache.db_path}')  # type: ignore
    else:
        session = requests.Session()

    url = pop_field(config, 'url')

    def open_url() -> IO:
        response = session.get(url, stream=True)
        fp = io.StringIO(response.text)
        return fp

    return _parse_feature_data_csv(config, message_prefix=f'{url}: ', open_file_fn=open_url)


@config_parser
def feature_data_csv_file_parser(config: ContextAwareDict, output_dir: Path) -> FeatureDataCsvParser:
    """
    Initialize the CSV parser that reads data from a stream.

    Parameters
    ----------
    config : ContextAwareDict
        Configuration mapping used to construct this component.
    output_dir : Path
        Directory where generated outputs are written.

    Returns
    -------
    FeatureDataCsvFileParser
        FeatureData object parsed from the source.
    """
    file = Path(pop_field(config, 'file'))
    file = search_path(file)
    open_file_fn = partial(open, file, 'r')
    return _parse_feature_data_csv(config, message_prefix=f'{file}: ', open_file_fn=open_file_fn)
