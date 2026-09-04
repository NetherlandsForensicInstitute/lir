from pathlib import Path

from lir import DataProvider, FeatureData
from lir.config import ConfigValue, config_parser, pop_field

from .sqlite_reader import read_from_sqlite3


class SqliteDataProvider(DataProvider):
    def __init__(self, path: str):
        self.path = path

    def get_instances(self) -> FeatureData:
        return read_from_sqlite3(self.path)


@config_parser
def parse_sqlite_data_provider_config(config: ConfigValue, output_dir: Path) -> SqliteDataProvider:
    path = pop_field(config, 'path')
    return SqliteDataProvider(path)
