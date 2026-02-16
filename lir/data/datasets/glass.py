import csv
from os import PathLike
from pathlib import Path

import numpy as np

from lir.data.data_strategies import RoleAssignment
from lir.data.io import RemoteResource
from lir.data.models import DataProvider, FeatureData


class GlassData(DataProvider):
    """
    LA-ICP-MS measurements of elemental concentration from floatglass.

    The measurements are from reference glass from casework, collected in the past 10 years or so.
    For more info on the DataProvider, see: https://github.com/NetherlandsForensicInstitute/elemental_composition_glass

    This data provider has a pre-defined train/test split, with a training set of three instances per source, and a test
    set of five instances per source.

    Data are retrieved from the web as needed and stored locally for later use.
    """

    def __init__(self, cache_dir: PathLike):
        self.resources = RemoteResource(
            'https://raw.githubusercontent.com/NetherlandsForensicInstitute/elemental_composition_glass/main',
            Path(cache_dir),
        )

    def _load_data(self, file: str, role: RoleAssignment) -> FeatureData:
        """Return a tuple of features, source_ids and instance_ids."""
        source_ids = []
        instance_ids = []
        values = []
        with self.resources.open(file, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                # the first measurement is at row 2, since row 1 is the header
                row_number = i + 2

                source_ids.append(f'{role.value}{row["Item"]}')
                instance_ids.append(f'{file}:{row_number}')
                values.append(np.array(list(map(float, row.values()))[3:]))

        return FeatureData(
            features=np.array(values),
            source_ids=np.array(source_ids),
            instance_ids=np.array(instance_ids),  # type: ignore
            role_assignments=np.array([role.value] * len(values)),  # type: ignore
        )

    def get_instances(self) -> FeatureData:
        """
        Return data with pre-defined assignments of training data and test data.

        The training data is read from `training.csv` and has three instances (replicates) per source.
        The test data is read from `duplo.csv` and `triplo.csv` and has a total of five instances per source.

        The features are elemental concentrations on a log_10 basis, and normalized to Si.
        The elements are: K39, Ti49, Mn55, Rb85, Sr88, Zr90, Ba137, La139, Ce140, Pb208

        The source_ids are unique identifiers of a glass particle. Each particle is from a different reference window.
        An instance is a replicate measurement on a glass particle. Source ids are prefixed with the role assignment,
        e.g. 'test-123' and 'train-123'. The ids 'test-123' and 'train-123' refer to different glass particles (and
        therefore different reference windows).

        The instance_ids values of an instance are a concatenation of the filename and a row number,
        e.g. "training.csv:22".

        The data are returned as a FeatureData object with the following properties:
        - features: an (n, 10) array of feature values
        - source_ids: a 1d array of source ids (str)
        - instance_ids: a 1d array of unique instance ids (str)
        - role_assignments: a 1d array of role assignments (values "train" or "test")
        """
        training_data = self._load_data('training.csv', RoleAssignment.TRAIN)
        duplo = self._load_data('duplo.csv', RoleAssignment.TEST)
        triplo = self._load_data('triplo.csv', RoleAssignment.TEST)

        return training_data + duplo + triplo
