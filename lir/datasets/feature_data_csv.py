import csv
import io
import itertools
import logging
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import IO, Any, NamedTuple

import numpy as np
import requests
from requests_cache import CachedSession

from lir.config.base import ContextAwareDict, config_parser, pop_field
from lir.data.io import search_path
from lir.data.models import DataProvider, FeatureData, InstanceData, NoData
from lir.data_strategies import RoleAssignment
from lir.util import check_type


LOG = logging.getLogger(__name__)


class ParseError(ValueError):
    """
    Exception to be raised on parse errors.

    This happens when an input file is malformatted or contains invalid input.
    """


class ExtraField(NamedTuple):
    """Extra field for CSV parsing."""

    field_name: str
    column_name: str
    validate_cell: Callable[[str], Any]

    def parse_row(self, row: dict[str, str]) -> list[Any]:
        """
        Take the appropriate values from a dictionary and return them as a list.

        Parameters
        ----------
        row : dict[str, str]
            CSV row dictionary to parse.

        Returns
        -------
        list[Any]
            Parsed values extracted from the input row.
        """
        values = []
        for colname in self.column_name:
            try:
                values.append(self.validate_cell(row[colname]))
            except Exception as e:
                raise ParseError(f'parsing value of column `{colname}` failed: {e}')
        return values


class CsvParser(DataProvider):
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
        self.source_id_columns: list[str] = (
            [source_id_column] if isinstance(source_id_column, str) else source_id_column or []
        )
        if len(self.source_id_columns) > 2:
            raise ValueError(f'the number of source id columns can be at most 2; found: {len(self.source_id_columns)}')

        self.label_column = label_column
        self.instance_id_column = instance_id_column
        self.role_assignment_column = role_assignment_column
        self.fold_assignment_column = fold_assignment_column
        self.extra_fields: list[ExtraField] = extra_fields or []
        self.ignore_columns: list[str] = ignore_columns or []
        self._head = head
        self._message_prefix = message_prefix
        self.continue_on_error = continue_on_error

        # the "extra field" argument allows for including arbitrary fields
        # check that they do not conflict with fields that are facilitated otherwise
        empty_data = FeatureData(features=np.ones((0, 1)))
        for field in extra_fields or []:
            if field.field_name in empty_data.all_fields:
                raise ValueError(
                    f'field {field.field_name} should not be read as an *extra* field; '
                    'use the appropriate config parameters instead'
                )

    def _initialize_reader(self, reader: csv.DictReader) -> None:
        """
        Initialize the CSV reader.

        This method is called immediately after a CSV reader has opened a file, and the field names have been read. The
        default behavior is to do nothing. It may be overridden by subclasses.

        Parameters
        ----------
        reader : csv.DictReader
            The CSV reader.
        """

    @staticmethod
    def _parse_value(line_num: int, column_name: str, value: str, parse_fn: Callable[[str], Any]) -> Any:
        try:
            return parse_fn(value)
        except (ValueError, TypeError) as e:
            raise ParseError(f'failed to parse {value} at row {line_num}, column {column_name}: {e}') from e

    def _parse_row(
        self, row: dict[str, str], reader: csv.DictReader
    ) -> dict[str, str | list[str] | int | float | list[float]]:
        fields: dict[str, str | list[str] | int | float | list[float]] = {}
        if self.source_id_columns:
            fields['source_ids'] = [row[column_name] for column_name in self.source_id_columns]
        if self.label_column is not None:
            fields['labels'] = self._parse_value(reader.line_num, self.label_column, row[self.label_column], int)
        if self.instance_id_column is not None:
            fields['instance_ids'] = row[self.instance_id_column]
        if self.role_assignment_column is not None:
            fields['role_assignments'] = self._parse_value(
                reader.line_num, self.role_assignment_column, row[self.role_assignment_column], RoleAssignment
            ).value
        if self.fold_assignment_column is not None:
            fields['fold_assignment_column'] = row[self.fold_assignment_column]
        for field in self.extra_fields:
            fields[field.field_name] = self._parse_value(
                reader.line_num, field.column_name, row[field.column_name], field.validate_cell
            )
        return fields

    def _assigned_columns(self, all_columns: Sequence[str]) -> list[tuple[str, str]]:
        """
        Obtain a list of columns that are explicitly assigned to an attribute.

        Parameters
        ----------
        all_columns : Sequence[str]
            A list of all available columns in the CSV file.

        Returns
        -------
        list[tuple[str, str]]
            A list of columns, with for each column a tuple (ROLE, COLUMN_NAME).
        """
        assigned_cols = [
            (role, column_name)
            for role, column_name in [
                ('label_column', self.label_column),
                ('instance_id_column', self.instance_id_column),
                ('role_assignment_column', self.role_assignment_column),
                ('fold_assignment_column', self.fold_assignment_column),
            ]
            if column_name is not None
        ]

        assigned_cols.extend(('source_id_column', column_name) for column_name in self.source_id_columns)
        assigned_cols.extend(('extra_fields', field.column_name) for field in self.extra_fields)
        assigned_cols.extend(('ignore_columns', column_name) for column_name in self.ignore_columns)
        return assigned_cols

    def _parse_file(self, fp: IO) -> InstanceData:
        # initialize the reader
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ParseError(f'{self._message_prefix}empty file')

        self._initialize_reader(reader)

        # check if all required columns exist in the csv file
        for name, value in self._assigned_columns(reader.fieldnames):
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
                    LOG.info(f'{self._message_prefix}parsing failed: {e}')
                else:
                    raise e

        if not all_instances:
            return NoData()

        if self._head is not None and n_instances < self._head:
            LOG.warning(f'input file has too few rows; expected: {self._head}; found: {n_instances}')

        # finalize the data
        all_instances = {k: np.array(v) for k, v in all_instances.items()}
        if self.fold_assignment_column and len(set(all_instances['fold_assignments'])) < 2:
            raise ParseError(
                f'{self._message_prefix}fold assignment column `{self.fold_assignment_column}` should contain at '
                f'least two different values; found: {set(all_instances["fold_assignments"])}'
            )

        return NoData(**all_instances)  # type: ignore

    def get_instances(self) -> InstanceData:
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


class FeatureDataCsvParser(CsvParser):
    """
    Parse a CSV file into an :class:`~lir.InstanceData` object.

    The CSV file contents are provided by the ``open_file_fn`` argument.

    Parameters
    ----------
    open_file_fn : Callable[[], IO]
        Function that returns a data stream from which the CSV file contents can be read.
    feature_columns : str | list[str] | None
        Column names containing numerical feature values. If not specified, all columns not otherwise designated are
        interpreted as feature columns.
    **kwargs : dict
        Other arguments are passed to the super class constructor.

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
        feature_columns: str | list[str] | None = None,
        **kwargs: dict,
    ):
        super().__init__(open_file_fn, **kwargs)  # type: ignore
        self.feature_columns: list[str] | None = (
            [feature_columns] if isinstance(feature_columns, str) else feature_columns
        )
        self._real_feature_columns: list[str] | None = None

    def _assigned_columns(self, all_columns: Sequence[str]) -> list[tuple[str, str]]:
        cols = super()._assigned_columns(all_columns)
        if self.feature_columns:
            cols.extend(('feature_columns', column_name) for column_name in self.feature_columns)

        return cols

    def _parse_row(
        self, row: dict[str, str], reader: csv.DictReader
    ) -> dict[str, str | list[str] | int | float | list[float]]:
        if self._real_feature_columns is None:
            if self.feature_columns:
                self._real_feature_columns = self.feature_columns
            else:
                # identify the feature columns
                fieldnames: list[str] = reader.fieldnames  # type: ignore
                explicit_role_column_names = {column_name for _, column_name in self._assigned_columns(fieldnames)}
                self._real_feature_columns = [
                    fieldname for fieldname in fieldnames if fieldname not in explicit_role_column_names
                ]

        fields = super()._parse_row(row, reader)
        fields['features'] = [
            self._parse_value(reader.line_num, fieldname, row[fieldname], float)
            for fieldname in self._real_feature_columns
        ]
        return fields

    def _parse_file(self, fp: IO) -> FeatureData:
        return super()._parse_file(fp).replace_as(FeatureData)


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

        return ExtraField(field_name, column_name, cell_type)

    raise ValueError(f'Extra field expected str or dict, but got {type(config)} ')


def _parse_feature_data_csv(config: ContextAwareDict, **kwargs: Any) -> FeatureDataCsvParser:
    extra_fields_config = pop_field(config, 'extra_fields', default=[], validate=partial(check_type, list))
    extra_fields = [_parse_extra_field(field_config) for field_config in extra_fields_config]

    parser = FeatureDataCsvParser(**config, extra_fields=extra_fields, **kwargs)  # type: ignore
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
