import sqlite3
import tempfile
from pathlib import Path

import numpy as np

from .sqlite_reader import read_from_sqlite3


def test_sqlite_reader():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = Path(tmpdir) / 'test.db'
        with sqlite3.connect(filename) as db:
            db.execute('CREATE TABLE feature_table(hypothesis INT, feature1 FLOAT, feature2 FLOAT)')
            db.execute(
                'INSERT INTO feature_table (hypothesis, feature1, feature2) VALUES (0, 1, 2), (0, 3, 4), (1, 5, 6)'
            )

        data = read_from_sqlite3(filename)
        assert np.all(data.features == np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 2))
