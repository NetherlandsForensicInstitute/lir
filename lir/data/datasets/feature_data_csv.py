import csv
import io
import logging
from abc import ABC
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import IO, Any, NamedTuple

import numpy as np
import requests
from requests_cache import CachedSession

from lir.config.base import ContextAwareDict, check_is_empty, config_parser, pop_field
from lir.data.data_strategies import RoleAssignment
from lir.data.io import search_path
from lir.data.models import DataProvider, FeatureData
from lir.util import check_type


LOG = logging.getLogger(__name__)


class ExtraField(NamedTuple):
    """Extra field for CSV parsing."""

    name: str
    column_names: list[str]
    validate_cell: Callable[[str], Any]

    def parse_row(self, row: dict[str, str]) -> list[Any]:
        """
        Take the appropriate values from a dictionary and return them as a list.

        :param row: the CSV line as a dictionary.
        :return: a list of values.
        """
        values = []
        for colname in self.column_names:
            try:
                values.append(self.validate_cell(row[colname]))
            except Exception as e:
                raise ValueError(f'parsing value of column `{colname}` failed: {e}')
        return values


class FeatureDataCsvParser(DataProvider, ABC):
    """
    Parses a csv file into a FeatureData object.

    This is an abstract class with implementations for different sources:
    - for reading from a local file, use `FeatureDataCsvFileParser`;
    - for reading from a URL, use `FeatureDataCsvHttpParser`;
    - for reading from a stream, use `FeatureDataCsvStreamParser`.

    Example: let's say we have data with two features and source ids.

    ```
    source_id,feature1,feature2,feature3,name_of_an_irrelevant_column
    0,1,10,1,sherlock
    0,1,11,1,holmes
    1,20,30,1,irene
    1,18,32,3,adler
    2,5,10,8,professor
    2,1,11,8,moriarty
    ```

    This file can be parsed from the following YAML:
    ```yaml
    data:
      provider: feature_data_csv
      path: path/to/file.csv
      source_id_column: source_id
      ignore_columns:
        - name_of_an_irrelevant_column
    ```
    """

    def __init__(
        self,
        source_id_column: str | list[str] | None = None,
        label_column: str | None = None,
        instance_id_column: str | None = None,
        role_assignment_column: str | None = None,
        extra_fields: list[ExtraField] | None = None,
        ignore_columns: list[str] | None = None,
        message_prefix: str = '',
    ):
        """
        Initializes the parser.

        Special columns can be assigned as such or can be ignored (see below). All other columns are interpreted as
        feature columns. All arguments are optional.

        :param source_id_column: the name (or list of two names) of the column that has the source ids
        :param label_column: the name of the column that contains the hypothesis label (0 or 1)
        :param instance_id_column: the name of the column that contains the instance id (str)
        :param role_assignment_column: the name of the column that contains the role assignment ("train" or "test")
        :param extra_fields: extra fields to read, in addition to the above
        :param ignore_columns: the names of the columns that should be ignored
        :param message_prefix: a string to prefix to all log and error messages
        """
        self.source_id_columns: list[str] = (
            [source_id_column] if isinstance(source_id_column, str) else source_id_column or []
        )
        if len(self.source_id_columns) > 2:
            raise ValueError(f'the number of source id columns can be at most 2; found: {len(self.source_id_columns)}')

        self.label_column = label_column
        self.instance_id_column = instance_id_column
        self.role_assignment_column = role_assignment_column
        self.extra_fields = extra_fields or []
        self.ignore_columns = ignore_columns or []
        self._message_prefix = message_prefix

        # the "extra field" argument allows for including arbitrary fields
        # check that they do not conflict with fields that are facilitated otherwise
        empty_data = FeatureData(features=np.ones((0, 1)))
        for field in extra_fields or []:
            if field.name in empty_data.all_fields:
                raise ValueError(
                    f'field {field.name} should not be read as an *extra* field; '
                    'use the appropriate config parameters instead'
                )

    def _parse_value(self, line_num: int, column_name: str, value: str, parse_fn: Callable[[str], Any]) -> Any:
        try:
            return parse_fn(value)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f'{self._message_prefix}failed to parse {value} at row {line_num}, column {column_name}: {e}'
            ) from e

    def _parse_file(self, fp: IO) -> FeatureData:
        # initialize the reader
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ValueError(f'{self._message_prefix}empty file')

        # identify the non feature columns
        non_feature_columns = (
            [
                ('label_column', self.label_column),
                ('instance_id_column', self.instance_id_column),
                ('role_assignment_column', self.role_assignment_column),
            ]
            + [('source_id_column', column_name) for column_name in self.source_id_columns]
            + [
                ('extra_fields', column_name)
                for column_name in chain.from_iterable([field.column_names for field in self.extra_fields])
            ]
            + [('ignore_columns', column_name) for column_name in self.ignore_columns]
        )

        # check if all required columns exist in the csv file
        for name, value in non_feature_columns:
            if value is not None and value not in reader.fieldnames:
                raise ValueError(
                    f'{self._message_prefix}{name} specified as `{value}`, but it is not present in the csv file'
                )

        # identify the feature columns
        non_feature_column_names = {column_name for _, column_name in non_feature_columns}
        feature_columns = [fieldname for fieldname in reader.fieldnames if fieldname not in non_feature_column_names]

        # initialize the result values
        source_ids: list[list[Any]] | None = [] if self.source_id_columns else None
        labels = []
        instance_ids = []
        role_assignments = []
        features = []
        extra_values = defaultdict(list)

        # read the file, row by row
        for row in reader:
            if source_ids is not None:
                source_ids.append([row[column_name] for column_name in self.source_id_columns])
            if self.label_column is not None:
                labels.append(self._parse_value(reader.line_num, self.label_column, row[self.label_column], int))
            if self.instance_id_column is not None:
                instance_ids.append(row[self.instance_id_column])
            if self.role_assignment_column is not None:
                role_assignments.append(
                    self._parse_value(
                        reader.line_num, self.role_assignment_column, row[self.role_assignment_column], RoleAssignment
                    ).value
                )
            for field in self.extra_fields:
                extra_values[field.name].append(field.parse_row(row))
            features.append(
                [self._parse_value(reader.line_num, fieldname, row[fieldname], float) for fieldname in feature_columns]
            )

        # finalize the data
        data = {
            'source_ids': np.array(source_ids) if source_ids else None,
            'labels': np.array(labels) if self.label_column is not None else None,
            'features': np.array(features),
        }
        data.update({k: np.array(v) for k, v in extra_values.items()})
        if self.instance_id_column is not None:
            data['instance_ids'] = np.array(instance_ids)
        if self.role_assignment_column is not None:
            data['role_assignments'] = np.array(role_assignments)

        return FeatureData(**data)  # type: ignore


class FeatureDataCsvFileParser(FeatureDataCsvParser):
    """Read CSV data from file."""

    def __init__(self, file: PathLike, **kwargs: Any):
        """
        Initializes the CSV parser that reads data from a file.

        :param file: the path to the csv file
        :param kwargs: arguments passed to FeatureDataCsvParser
        """
        super().__init__(**kwargs, message_prefix=f'{file}: ')
        self.path = Path(file)

    def get_instances(self) -> FeatureData:
        """Retrieve FeatureData instances."""
        path = search_path(self.path)
        LOG.debug(f'parsing CSV file: {self.path} as {path}')
        with open(path) as f:
            return self._parse_file(f)


class FeatureDataCsvStreamParser(FeatureDataCsvParser):
    """Read data from a streamed CSV."""

    def __init__(self, fp: IO, **kwargs: Any):
        """
        Initializes the CSV parser that reads data from a stream.

        :param flo: the file-like object
        :param kwargs: arguments passed to FeatureDataCsvParser
        """
        super().__init__(**kwargs)
        self.fp = fp

    def get_instances(self) -> FeatureData:
        """Retrieve FeatureData instances from CSV stream."""
        LOG.debug('parsing CSV stream')
        return self._parse_file(self.fp)


class FeatureDataCsvHttpParser(FeatureDataCsvParser):
    """Read data from a stream."""

    def __init__(self, url: str, session: requests.Session, **kwargs: Any):
        """
        Initializes the CSV parser that reads data from a stream.

        By default, this class uses `requests-cache` to cache retrieved data. The cache is persistent and located in
        the user cache folder, which is written to the log file.

        :param url: the URL to the CSV resource
        :param session: requests session for HTTP requests
        :param kwargs: arguments passed to FeatureDataCsvParser
        """
        super().__init__(**kwargs, message_prefix=url)
        self.url = url
        self.session = session

    def get_instances(self) -> FeatureData:
        """Retrieve FeatureData from the remote resource."""
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


def _parse_extra_field(config: ContextAwareDict) -> ExtraField:
    name = pop_field(config, 'name')
    columns = pop_field(config, 'columns', validate=partial(check_type, list))
    cell_type = pop_field(config, 'cell_type', validate=_parse_cell_type)
    check_is_empty(config)
    return ExtraField(name, columns, cell_type)


def _parse_feature_data_csv(
    parser_class: type[FeatureDataCsvParser], config: ContextAwareDict, **kwargs: Any
) -> FeatureDataCsvParser:
    extra_fields_config = pop_field(config, 'extra_fields', default=[], validate=partial(check_type, list))
    extra_fields = [_parse_extra_field(field_config) for field_config in extra_fields_config]

    parser = parser_class(**config, extra_fields=extra_fields, **kwargs)
    return parser


@config_parser
def feature_data_csv_http_parser(config: ContextAwareDict, output_dir: Path) -> FeatureDataCsvParser:
    """
    Configuration parser to initialize the CSV parser that reads data from a stream.

    Arguments:
        - use_cache: boolean indicating whether to cache retrieved data

    Other arguments are passed directly to `FeatureDataCsvParser`.
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
def feature_data_csv_file_parser(config: ContextAwareDict, output_dir: Path) -> FeatureDataCsvParser:
    """Configuration parser to initialize the CSV parser that reads data from a stream."""
    return _parse_feature_data_csv(FeatureDataCsvFileParser, config)
