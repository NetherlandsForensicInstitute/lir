import tempfile
from pathlib import Path
from typing import Any

import confidence
import pytest

from lir.config.substitution import FolderHyperparameter, HyperparameterOption, _expand, parse_hyperparameter
from lir.data.io import search_path


@pytest.mark.parametrize(
    'desc,yaml,expected_options',
    [
        pytest.param(
            'categorical with simple options',
            """
                path: "param.root"
                options:
                  - a
                  - b
            """,
            [
                HyperparameterOption('a', {'param.root': 'a'}),
                HyperparameterOption('b', {'param.root': 'b'}),
            ],
        ),  # pytest.param
        pytest.param(
            'categorical with named options',
            """
                path: "param.root"
                options:
                  - option_name: option_a
                    value: a
                  - option_name: option_b
                    value: b
            """,
            [
                HyperparameterOption('option_a', {'param.root': 'a'}),
                HyperparameterOption('option_b', {'param.root': 'b'}),
            ],
        ),  # pytest.param
        pytest.param(
            'clustered',
            """
                name: "hieperdepieper"
                options:
                  - option_name: option_a
                    substitutions:
                      - path: param.root
                        value: a
                  - option_name: option_b
                    substitutions:
                      - path: param.root
                        value: b
            """,
            [
                HyperparameterOption('option_a', {'param.root': 'a'}),
                HyperparameterOption('option_b', {'param.root': 'b'}),
            ],
        ),  # pytest.param
        pytest.param(
            'constant',
            """
                path: "param.root"
                value: a
            """,
            [
                HyperparameterOption('a', {'param.root': 'a'}),
            ],
        ),  # pytest.param
        pytest.param(
            'numerical linear',
            """
                path: "param.root"
                low: -1
                high: 1
                step: 1
            """,
            [
                HyperparameterOption('-1', {'param.root': -1}),
                HyperparameterOption('0', {'param.root': 0}),
                HyperparameterOption('1', {'param.root': 1}),
            ],
        ),  # pytest.param
    ],
)
def test_substitution(desc, yaml: str, expected_options: list[Any]):
    cfg = _expand([], confidence.loads(yaml))
    param = parse_hyperparameter(cfg, Path('/'))
    assert param.options() == expected_options, desc


def test_folder_hyperparameter():
    tmp_folder = tempfile.mkdtemp()
    for i in range(3):
        (Path(tmp_folder) / f'file_{i}.txt').touch()
    folders = FolderHyperparameter('name', str(tmp_folder))

    # We expect three files in the temporary folder
    assert len(folders.options()) == 3

    # Check that the names correspond to the created files
    expected_names = {str(search_path(Path(tmp_folder) / f'file_{i}.txt')) for i in range(3)}
    actual_names = {list(opt.substitutions.values())[0] for opt in folders.options()}
    assert actual_names == expected_names


def test_folder_hyperparameter_ignore():
    tmp_folder = tempfile.mkdtemp()
    for i in range(3):
        (Path(tmp_folder) / f'file_{i}.txt').touch()
    folders = FolderHyperparameter('name', str(tmp_folder), ignore_files=['*1.txt'])

    # We expect two files in the temporary folder, as file_1.txt is ignored.
    assert len(folders.options()) == 2

    # Check that the names correspond to the created files
    expected_names = {str(search_path(Path(tmp_folder) / f'file_{i}.txt')) for i in (0, 2)}
    actual_names = {list(opt.substitutions.values())[0] for opt in folders.options()}
    assert actual_names == expected_names


def test_folder_hyperparameter_value_errors():
    with pytest.raises(ValueError):
        FolderHyperparameter('name', '/path/does/not/exist')

    tmp_folder = tempfile.mkdtemp()
    FolderHyperparameter('name', tmp_folder)
    with pytest.raises(ValueError):
        FolderHyperparameter('name', tmp_folder).options()
