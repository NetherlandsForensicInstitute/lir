from pathlib import Path
from typing import Any

import confidence
import pytest

from lir.config.substitution import HyperparameterOption, _expand, parse_hyperparameter


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
