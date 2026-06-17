import csv
import io
import itertools
import logging
from abc import ABC
from collections.abc import Callable
from functools import partial
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import IO, Any, NamedTuple

import numpy as np
import requests
from requests_cache import CachedSession

from lir.config.base import ContextAwareDict, config_parser, pop_field
from lir.data.io import search_path
from lir.data.models import DataProvider, FeatureData
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

    name: str
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


class FeatureDataCsvParser(DataProvider, ABC):
    """
    Parse a CSV file into a ``FeatureData`` object.

    This is an abstract base class with concrete implementations for different
    data sources:

    - :class:`FeatureDataCsvFileParser` for reading from a local file;
    - :class:`FeatureDataCsvHttpParser` for reading from a URL;
    - :class:`FeatureDataCsvStreamParser` for reading from a stream.

    Parameters
    ----------
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
        self.source_id_columns: list[str] = (
            [source_id_column] if isinstance(source_id_column, str) else source_id_column or []
        )
        if len(self.source_id_columns) > 2:
            raise ValueError(f'the number of source id columns can be at most 2; found: {len(self.source_id_columns)}')

        self.label_column = label_column
        self.instance_id_column = instance_id_column
        self.feature_columns: list[str] | None = (
            [feature_columns] if isinstance(feature_columns, str) else feature_columns
        )
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
            if field.name in empty_data.all_fields:
                raise ValueError(
                    f'field {field.name} should not be read as an *extra* field; '
                    'use the appropriate config parameters instead'
                )

    @staticmethod
    def _parse_value(line_num: int, column_name: str, value: str, parse_fn: Callable[[str], Any]) -> Any:
        try:
            return parse_fn(value)
        except (ValueError, TypeError) as e:
            raise ParseError(f'failed to parse {value} at row {line_num}, column {column_name}: {e}') from e

    def _parse_row(
        self, row: dict[str, str], reader: csv.DictReader, feature_columns: list[str]
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
            fields[field.name] = field.parse_row(row)
        fields['features'] = [
            self._parse_value(reader.line_num, fieldname, row[fieldname], float) for fieldname in feature_columns
        ]
        return fields

    def _parse_file(self, fp: IO) -> FeatureData:
        # initialize the reader
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ParseError(f'{self._message_prefix}empty file')

        # identify the non feature columns
        columns_with_explicit_role = (
            [
                ('label_column', self.label_column),
                ('instance_id_column', self.instance_id_column),
                ('role_assignment_column', self.role_assignment_column),
                ('fold_assignment_column', self.fold_assignment_column),
            ]
            + [('source_id_column', column_name) for column_name in self.source_id_columns]
            + [
                ('extra_fields', column_name)
                for column_name in chain.from_iterable([field.column_name for field in self.extra_fields])
            ]
            + [('ignore_columns', column_name) for column_name in self.ignore_columns]
        )

        if self.feature_columns:
            columns_with_explicit_role.extend(('feature_columns', column_name) for column_name in self.feature_columns)
            feature_columns = self.feature_columns
        else:
            # identify the feature columns
            explicit_role_column_names = {column_name for _, column_name in columns_with_explicit_role}
            feature_columns = [
                fieldname for fieldname in reader.fieldnames if fieldname not in explicit_role_column_names
            ]

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
                fields = self._parse_row(row, reader, feature_columns)
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
            return FeatureData(features=np.zeros((0, 1)))

        if self._head is not None and n_instances < self._head:
            LOG.warning(f'input file has too few rows; expected: {self._head}; found: {n_instances}')

        # finalize the data
        all_instances = {k: np.array(v) for k, v in all_instances.items()}
        if self.fold_assignment_column and len(set(all_instances['fold_assignments'])) < 2:
            raise ParseError(
                f'{self._message_prefix}fold assignment column `{self.fold_assignment_column}` should contain at '
                f'least two different values; found: {set(all_instances["fold_assignments"])}'
            )

        return FeatureData(**all_instances)  # type: ignore


class FeatureDataCsvFileParser(FeatureDataCsvParser):
    """
    Read CSV data from file.

    Parameters
    ----------
    file : PathLike
        Path to the input file.
    **kwargs : Any
        Additional keyword arguments forwarded to the underlying FeatureDataCsvParser call.
    """

    def __init__(self, file: PathLike, **kwargs: Any):
        super().__init__(**kwargs, message_prefix=f'{file}: ')
        self.path = Path(file)

    def get_instances(self) -> FeatureData:
        """
        Retrieve FeatureData instances.

        Returns
        -------
        FeatureData
            FeatureData object parsed from the source.
        """
        path = search_path(self.path)
        LOG.debug(f'parsing CSV file: {self.path} as {path}')
        with open(path) as f:
            return self._parse_file(f)


class FeatureDataCsvStreamParser(FeatureDataCsvParser):
    """
    Read data from a streamed CSV.

    Parameters
    ----------
    fp : IO
        Open file-like object to read from.
    **kwargs : Any
        Additional keyword arguments forwarded to the underlying call.
    """

    def __init__(self, fp: IO, **kwargs: Any):
        super().__init__(**kwargs)
        self.fp = fp

    def get_instances(self) -> FeatureData:
        """
        Retrieve FeatureData instances from CSV stream.

        Returns
        -------
        FeatureData
            FeatureData object parsed from the source.
        """
        LOG.debug('parsing CSV stream')
        return self._parse_file(self.fp)


class FeatureDataCsvHttpParser(FeatureDataCsvParser):
    """
    Read CSV data from a URL.

    By default, this class uses `requests-cache` to cache retrieved data. The cache is persistent and located in
    the user cache folder, which is written to the log file.

    Parameters
    ----------
    url : str
        URL of the remote resource to read.
    session : requests.Session
        Value passed via ``session``.
    **kwargs : Any
        Additional keyword arguments forwarded to the underlying call.
    """

    def __init__(self, url: str, session: requests.Session, **kwargs: Any):
        super().__init__(**kwargs, message_prefix=f'{url}: ')
        self.url = url
        self.session = session

    def get_instances(self) -> FeatureData:
        """
        Retrieve FeatureData from the remote resource.

        Returns
        -------
        FeatureData
            FeatureData object parsed from the source.
        """
        LOG.debug(f'parsing CSV from URL: {self.url}')
        response = self.session.get(self.url, stream=True)
        fp = io.StringIO(response.text)
        return self._parse_file(fp)


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
    if isinstance(config, str):
        return ExtraField(config, config, str)

    if isinstance(config, ContextAwareDict):
        # It should only be a key/value pair
        if len(config.keys()) != 1:
            raise ValueError

        name = list(config.keys())[0]
        column_name = config[name]

        return ExtraField(name, column_name, str)

    raise ValueError(f'Extra field expected str or dict, but got {type(config)} ')


def _parse_feature_data_csv[ParserType: FeatureDataCsvParser](
    parser_class: type[ParserType], config: ContextAwareDict, **kwargs: Any
) -> ParserType:
    extra_fields_config = pop_field(config, 'extra_fields', default=[])
    extra_fields = [_parse_extra_field(field_config) for field_config in extra_fields_config]

    parser = parser_class(**config, extra_fields=extra_fields, **kwargs)
    return parser


@config_parser
def feature_data_csv_http_parser(config: ContextAwareDict, output_dir: Path) -> FeatureDataCsvHttpParser:
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

    return _parse_feature_data_csv(FeatureDataCsvHttpParser, config, session=session)


@config_parser
def feature_data_csv_file_parser(config: ContextAwareDict, output_dir: Path) -> FeatureDataCsvFileParser:
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
    return _parse_feature_data_csv(FeatureDataCsvFileParser, config)
