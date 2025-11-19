import datetime
import getpass
import logging
from pathlib import Path
from typing import Any

import git
import numpy as np
import pytz
from git import InvalidGitRepositoryError

from lir.aggregation import Aggregation


LOG = logging.getLogger(__name__)


class DumpGitInfo(Aggregation):
    def __init__(self, path: Path):
        self.path = path

    def report(self, llrs: np.ndarray, labels: np.ndarray | None, parameters: dict[str, Any]) -> None:
        pass

    def close(self) -> None:  # noqa: B027
        with open(self.path, 'w') as f:
            try:
                repo = git.Repo(search_parent_directories=True)
                if repo.is_dirty():
                    LOG.warning('working from a dirty git repository')
                    f.write('repository status: DIRTY\n')
                else:
                    f.write(f'repository status: {repo.head.object.hexsha}\n')
                    f.write(f'commit_time: {repo.head.object.committed_datetime.isoformat()}\n')
            except InvalidGitRepositoryError:
                f.write('repository status: UNAVAILABLE')

            f.write(f'run_time: {datetime.datetime.now(tz=pytz.utc).isoformat()}\n')
            f.write(f'user: {getpass.getuser()}\n')
