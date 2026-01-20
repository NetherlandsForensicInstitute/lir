import csv
import io
import logging
from abc import ABC
from collections.abc import Callable
from functools import partial
from os import PathLike
from pathlib import Path
from typing import IO, Any

import numpy as np
import requests
from requests_cache import CachedSession

from lir.config.base import ContextAwareDict, config_parser, pop_field
from lir.data.data_strategies import RoleAssignment
from lir.data.io import search_path
from lir.data.models import DataProvider, FeatureData
from lir.util import check_type


LOG = logging.getLogger(__name__)


class FeatureDataCsvParser(DataProvider, ABC):
    """
    Parses a csv file into a FeatureData object.

    This is an abstract class with implementations for different sources:
    - for reading from a local file, use `FeatureDataCsvFileParser`;
    - for reading from a URL, use `FeatureDataCsvHttpParser`;
    - for reading from a stream, use `FeatureDataCsvStreamParser`.

    Example: let's say we have data with two features and source ids.

    ```csv
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
        source_id_column: str | None = None,
        label_column: str | None = None,
        instance_id_column: str | None = None,
        role_assignment_column: str | None = None,
        ignore_columns: list[str] | None = None,
        message_prefix: str = '',
    ):
        """
        Initializes the parser.

        Special columns can be assigned as such (see below). All other columns are interpreted as feature columns.

        :param source_id_column: the name of the column that has the source ids (str)
        :param label_column: the name of the column that contains the hypothesis label (0 or 1)
        :param instance_id_column: the name of the column that contains the instance id (str)
        :param role_assignment_column: the name of the column that contains the role assignment ("train" or "test")
        :param ignore_columns: the names of the columns that should be ignored
        :param message_prefix: a string to prefix to all log and error messages
        """
        self.source_id_column = source_id_column
        self.label_column = label_column
        self.instance_id_column = instance_id_column
        self.role_assignment_column = role_assignment_column
        self.ignore_columns = ignore_columns or []
        self._message_prefix = message_prefix

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

        # check if all required columns exist in the csv file
        for name, value in [
            ('source_id_column', self.source_id_column),
            ('label_column', self.label_column),
            ('instance_id_column', self.instance_id_column),
            ('role_assignment_column', self.role_assignment_column),
        ]:
            if value is not None and value not in reader.fieldnames:
                raise ValueError(
                    f'{self._message_prefix}{name} specified as `{value}`, but it is not present in the csv file'
                )

        # identify the feature columns
        special_columns = [
            self.source_id_column,
            self.label_column,
            self.instance_id_column,
            self.role_assignment_column,
        ] + self.ignore_columns
        feature_columns = [fieldname for fieldname in reader.fieldnames if fieldname not in special_columns]

        # initialize the result values
        source_ids = []
        labels = []
        instance_ids = []
        role_assignments = []
        features = []

        # read the file, row by row
        for row in reader:
            if self.source_id_column is not None:
                source_ids.append(row[self.source_id_column])
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
            features.append(
                [self._parse_value(reader.line_num, fieldname, row[fieldname], float) for fieldname in feature_columns]
            )

        # finalize the data
        data = {
            'source_ids': np.array(source_ids) if self.source_id_column is not None else None,
            'labels': np.array(labels) if self.label_column is not None else None,
            'features': np.array(features),
        }
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


@config_parser
def feature_data_csv_http_parser(config: ContextAwareDict, output_dir: Path) -> FeatureDataCsvHttpParser:
    """
    Configuration parser to initialize the CSV parser that reads data from a stream.

    Arguments:
        - use_cache: boolean indicating whether to cache retrieved data

    Other arguments are passed directly to `FeatureDataCsvParser`.
    ```
    """
    use_cache = pop_field(config, 'use_cache', default=True, validate=partial(check_type, bool))

    session: requests.Session
    if use_cache:
        session = CachedSession('lir', use_cache_dir=True)
        LOG.debug(f'using cache location: {session.cache.db_path}')  # type: ignore
    else:
        session = requests.Session()

    parser = FeatureDataCsvHttpParser(session=session, **config)
    return parser
