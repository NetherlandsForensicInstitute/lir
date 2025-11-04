import logging
import urllib.request
from pathlib import Path
from typing import IO, Any


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
