from pathlib import Path

from lir import DataProvider, FeatureData
from lir.config import ConfigValue, config_parser

from .sqlite_reader import read_from_sqlite3


class SqliteDataProvider(DataProvider):
    def __init__(self, path: str):
        self.path = path

    def get_instances(self) -> FeatureData:
        return read_from_sqlite3(self.path)


@config_parser
def parse_sqlite_data_provider_config(config: ConfigValue, output_dir: Path) -> SqliteDataProvider:
    # the use of `with` is optional, and adds a check that all fields in `config` are consumed
    with config:
        path = config.pop_field('path')
        return SqliteDataProvider(path)
