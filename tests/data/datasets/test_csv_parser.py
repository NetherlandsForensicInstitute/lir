import io
from pathlib import Path

import numpy as np
import pytest

from lir.data.data_strategies import RoleAssignment
from lir.data.datasets.feature_data_csv import FeatureDataCsvParser
from lir.data.models import FeatureData


@pytest.mark.parametrize(
    "file_contents,parser_args,expected_result,description",
    [
        ("", {}, None, "empty data"),
        (
                "feature1,feature2\n1,1\n",
                {},
                FeatureData(features=np.ones((1, 2))),
                "1 row, 2 features",
        ),
        (
                "feature1,feature2\n1,1\n1\n",
                {},
                None,
                "incomplete row",
        ),
        (
                "feature1,feature2\na,1\n",
                {},
                None,
                "non-numeric feature",
        ),
        (
                "feature1,feature2\n0,1\n2,3\n",
                {},
                FeatureData(features=np.arange(4).reshape(2, 2)),
                "2 rows, 2 features",
        ),
        (
                "feature1,feature2\n1,1\n",
                {"label_column": "label"},
                None,
                "label_column refers to non-existent column",
        ),
        (
                "label,feature1,feature2\n1,1,1\n",
                {"label_column": "label"},
                FeatureData(labels=np.array([1]), features=np.ones((1, 2))),
                "1 row, 2 features, with label",
        ),
        (
                "label,feature1,feature2\n2,1,1\n",
                {"label_column": "label"},
                None,
                "1 row, 2 features, bad label",
        ),
        (
                "label,source_id,feature1,feature2\n0,10,1,1\n",
                {"label_column": "label", "source_id_column": "source_id"},
                FeatureData(source_ids=np.array(["10"]), labels=np.array([0]), features=np.ones((1, 2))),
                "1 row, 2 features, with label, source_id",
        ),
        (
                "source_id,instance_id,feature1,feature2\n10,11,1,1\n",
                {"instance_id_column": "instance_id", "source_id_column": "source_id"},
                FeatureData(source_ids=np.array(["10"]), instance_ids=np.array(["11"]), features=np.ones((1, 2))),
                "1 row, 2 features, with source_id, instance_id",
        ),
        (
                "feature1,feature2,role\n0,1,train\n2,3,test\n",
                {"role_assignment_column": "role"},
                FeatureData(features=np.arange(4).reshape(2, 2), role_assignments=np.array([RoleAssignment.TRAIN.value, RoleAssignment.TEST.value])),
                "2 rows, 2 features, with roles",
        ),
        (
                "feature1,feature2,role\n0,1,train\n2,3,validate\n",
                {"role_assignment_column": "role"},
                None,
                "2 rows, 2 features, with bad role",
        ),
        (
                "feature1,feature2,feature3,feature4\n1,2,3,4\n",
                {"ignore_columns": ["feature1","feature4"]},
                FeatureData(features=np.array([[2, 3]])),
                "1 row, 2 features, 2 ignored",
        ),
    ]
)
def test_csv_parser(file_contents: str, parser_args: dict[str, str], expected_result: FeatureData, description: str):
    f = io.StringIO(file_contents)
    try:
        parser = FeatureDataCsvParser(path=f, **parser_args)
        actual_result = parser.get_instances()
        if expected_result is not None:
            assert actual_result == expected_result
        else:
            pytest.fail(f"'{description}' succeeded, but should raise an error")
    except Exception as e:
        if expected_result is not None:
            pytest.fail(f"while parsing '{description}': {e}", e)


@pytest.mark.parametrize("path,expected_full_path", [
    ("tests", Path(__file__).parent.parent.parent.parent / "tests"),
    ("lir/config", Path(__file__).parent.parent.parent.parent / "lir/config"),
    ("lir/non_existent", "lir/non_existent"),
])
def test_search_path(path: str, expected_full_path: str):
    full_path = FeatureDataCsvParser._search_path(Path(path))
    assert full_path == Path(expected_full_path).resolve()
