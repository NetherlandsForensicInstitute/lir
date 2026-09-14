import sqlite3

import numpy as np

from lir import FeatureData
from lir.config.data import data_provider


@data_provider
def read_from_sqlite3(path: str) -> FeatureData:
    with sqlite3.connect(path) as db:
        hypotheses = []
        features = []

        result = db.execute('SELECT hypothesis, feature1, feature2 FROM feature_table')
        for row in result:
            hypotheses.append(row[0])
            features.append([row[1], row[2]])

        return FeatureData(hypthesis=np.array(hypotheses), features=np.array(features))
