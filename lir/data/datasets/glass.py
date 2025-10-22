import csv
from os import PathLike
from pathlib import Path
from typing import Tuple, Iterator

import numpy as np

from lir.data.io import RemoteResource
from lir.data.models import DataSet, DataStrategy


class GlassData(DataSet, DataStrategy):
    """
    LA-ICP-MS measurements of elemental concentration from floatglass.

    The measurements are from reference glass from casework, collected in the past 10 years or so.
    For more info on the dataset, see: https://github.com/NetherlandsForensicInstitute/elemental_composition_glass

    This class is a `DataSet` as well as a `DataStrategy`, as it implements both `get_instances()` and `__iter__()`.
    When used as a `DataSet`, it uses `get_instances()` which returns a dataset of three instances per source.
    When used as a `DataStrategy`, it uses `__iter__()` which returns a training/test set combination. In that case,
    the training set is identical to the data returned by `get_instances()`. The test set has a total of five instances
    per source.

    Data are retrieved from the web as needed and stored locally for later use.
    """

    def __init__(self, cache_dir: PathLike):
        self.resources = RemoteResource(
            "https://raw.githubusercontent.com/NetherlandsForensicInstitute/elemental_composition_glass/main",
            Path(cache_dir),
        )

    def _load_data(self, file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns a tuple of instances, labels and meta data.
        """
        labels = []
        origins = []
        values = []
        with self.resources.open(file, "r") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                # the first measurement is at row 2, since row 1 is the header
                row_number = i + 2

                labels.append(int(row["Item"]))
                origins.append(f"{file}:{row_number}")
                values.append(np.array(list(map(float, row.values()))[3:]))

        return np.array(values), np.array(labels), np.array(origins).reshape(-1, 1)

    def __iter__(self) -> Iterator:
        """
        Returns an iterator over a single combination of training data and test data.

        The training data is read from `training.csv` and has three instances (replicates) per source.
        The test data is read from `duplo.csv` and `triplo.csv` and has a total of five instances per source.

        The data are returned as a tuple of features, labels, and metadata, similar to `get_instances()`.
        """
        training_data = self._load_data("training.csv")
        test_features, test_labels, test_meta = zip(
            self._load_data("duplo.csv"), self._load_data("triplo.csv")
        )
        test_data = (
            np.concatenate(test_features),
            np.concatenate(test_labels),
            np.concatenate(test_meta),
        )
        return iter([(training_data, test_data)])

    def get_instances(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns `training.csv` data with three instances (replicates) per source, as a tuple of features, labels and
        metadata.

        The features are elemental concentrations on a log_10 basis, and normalized to Si.
        The elements are: K39, Ti49, Mn55, Rb85, Sr88, Zr90, Ba137, La139, Ce140, Pb208

        The labels are unique identifiers of a glass particle. Each particle is from a different reference window. An
        instance is a replicate measurement on a glass particle.

        The meta values of an instance are a concatenation of the filename and a row number, e.g. "training.csv:22".
        """
        return self._load_data("training.csv")
