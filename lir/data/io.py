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

    @staticmethod
    def _get_headers(prefix: str, dimensions: list[int | list[str]]) -> list[str]:
        """
        Generate a one-dimensional list of headers for multi-dimensional data.

        The length of the returned list is equal to the product of the dimensions.

        The value of each element corresponds to the coordinates in the data.

        The values of `dimensions` may be a mixture of `int` and `list[str]` types. If any value is an `int`, it is the
        length of the dimension. If it is `list[str]`, it is a meaningful header for the dimension and its length is
        equal to the size of the dimension.

        :param prefix: prefix for all headers
        :param dimensions: dimension of each row in the data
        :return: a flattened array of headers
        """
        full_shape = [d if isinstance(d, int) else len(d) for d in dimensions]
        full_header = np.full(shape=full_shape, fill_value=prefix)

        for dim, size in enumerate(full_shape):
            # initialize the axis header
            shape = [size if dim == i else 1 for i in range(len(dimensions))]

            # build the axis header
            if isinstance(dimensions[dim], int):
                if size == 1:
                    header_indexes = np.full(shape=shape, fill_value='')
                else:
                    text_width = np.floor(np.log10(size - 1)) + 1
                    header_indexes = np.char.mod(f'.%0{text_width}d', np.arange(size)).reshape(*shape)
            else:
                header_indexes = np.array(dimensions[dim]).reshape(*shape)

            # extend the index header to all dimensions
            for broadcast_dimension, broadcast_size in enumerate(full_shape):
                if broadcast_dimension != dim:
                    header_indexes = np.repeat(header_indexes, broadcast_size, axis=broadcast_dimension)

            # append to the full header
            full_header = np.char.add(full_header, header_indexes)

        return list(full_header.reshape(np.prod(full_shape)))

    def add_column(
        self, data: np.ndarray, header_prefix: str = '', dimension_headers: dict[int, list[str]] | None = None
    ) -> None:
        """
        Append data and corresponding headers to `self._all_data` and `self._all_headers`.

        The data argument is an arbitrary numpy array. Its first dimension are the rows. Any other dimension will be
        columns in the CSV output.

        :param header_prefix: the prefix for all headers
        :param dimension_headers: a mapping from dimensions to its headers; the dimension corresponds to the dimensions
            of the data. Because dimension 0 corresponds to rows, it should have no headers
        :param data:
        """
        dimension_headers = dimension_headers or {}

        if len(data.shape) < 2:
            data = data.reshape(-1, 1)

        for dim, header in dimension_headers.items():
            if dim == 0 or dim >= len(data.shape):
                raise ValueError(f'dimension index out of bounds: {dim}')
            if len(header) != data.shape[dim]:
                raise ValueError(
                    f'the header size does not match the number of columns: {len(header)} != {data.shape[1]}'
                )

        dimensions = [dimension_headers.get(dim + 1, size) for dim, size in enumerate(data.shape[1:])]
        self._all_headers.extend(self._get_headers(header_prefix or '', dimensions))

        if len(data.shape) != 2:
            data = data.reshape(data.shape[0], -1)

        self._all_data.append(data)

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
