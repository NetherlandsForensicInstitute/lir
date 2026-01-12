import csv
import logging
import sys
import urllib.request
from itertools import chain
from pathlib import Path
from typing import IO, Any

import numpy as np


LOG = logging.getLogger(__name__)


class RemoteResource:
    def __init__(self, url: str, local_directory: Path):
        self.url = url
        self.local_directory = local_directory

    def open(self, filename: str, mode: str = 'r') -> IO[Any]:
        local_path = self.local_directory / filename
        if not local_path.exists():
            url = f'{self.url}/{filename}'
            local_path.parent.mkdir(exist_ok=True, parents=True)
            urllib.request.urlretrieve(url, local_path)

        LOG.debug(f'open file: {local_path}')
        return open(local_path, mode)


class DataFileBuilderCsv:
    def __init__(self, path: Path, write_mode: str = 'w', write_header: bool | None = None):
        self.path = path
        self.write_mode = write_mode
        self.write_header = write_header
        self._all_headers: list[str] = []
        self._all_data: list[np.ndarray] = []

    def add_column(self, header: str | list[str], data: np.ndarray) -> None:
        """
        Append data and corresponding headers to `self._all_data` and `self._all_headers`.
        """
        if len(data.shape) != 2:
            data = data.reshape(data.shape[0], -1)
        self._all_data.append(data)

        if isinstance(header, str):
            if data.shape[1] == 1:
                self._all_headers.append(header)
            else:
                self._all_headers.extend([f'{header}{i}' for i in range(data.shape[1])])
        elif isinstance(header, list):
            if len(header) == data.shape[1]:
                self._all_headers.extend(header)
            else:
                raise ValueError(
                    f'the header size does not match the number of columns: {len(header)} != {data.shape[1]}'
                )
        else:
            raise TypeError('argument `header` must be either `str` or `list[str]`')

    def write(self) -> None:
        LOG.info(f'writing CSV file: {self.path}')
        self.path.parent.mkdir(exist_ok=True, parents=True)

        write_header = self.write_header
        if write_header is None:
            write_header = self.write_mode == 'w' or not self.path.exists()

        with open(self.path, self.write_mode, newline='') as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow(self._all_headers)

            for row in zip(*self._all_data, strict=True):
                writer.writerow(chain(*row))


def search_path(path: Path) -> Path:
    """
    Searches the python path for a file.

    If `path` is absolute, it is normalized by `Path.resolve()` and returned.

    If `path` is relative, the file is searched in `sys.path`. The path is interpreted as relative to `sys.path`
    elements one by one, and if it exists, it is normalized by `Path.resolve()` and returned.

    If the file is not found, it is normalized and made absolute by `Path.resolve()` and returned.
    """
    if path.is_absolute():
        return path.resolve()

    for search_path_element in sys.path:
        full_path = Path(search_path_element) / path
        if full_path.exists():
            return full_path.resolve()

    return path.resolve()
