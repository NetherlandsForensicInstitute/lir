from pathlib import Path

import pytest

from lir.data.io import search_path


@pytest.mark.parametrize("path,expected_full_path", [
    ("tests", Path(__file__).parent.parent.parent / "tests"),
    ("lir/config", Path(__file__).parent.parent.parent / "lir/config"),
    ("lir/non_existent", "lir/non_existent"),
])
def test_search_path(path: str, expected_full_path: Path | str):
    full_path = search_path(Path(path))
    assert full_path == Path(expected_full_path).resolve()
