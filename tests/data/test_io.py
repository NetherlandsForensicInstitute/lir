from pathlib import Path

import numpy as np
import pytest

from lir.data.io import DataFileBuilderCsv, search_path


@pytest.mark.parametrize(
    'path,expected_full_path',
    [
        ('tests', Path(__file__).parent.parent.parent / 'tests'),
        ('lir/config', Path(__file__).parent.parent.parent / 'lir/config'),
        ('lir/non_existent', 'lir/non_existent'),
    ],
)
def test_search_path(path: str, expected_full_path: Path | str):
    full_path = search_path(Path(path))
    assert full_path == Path(expected_full_path).resolve()


@pytest.mark.parametrize(
    'prefix,dimensions,expected_output,msg',
    [
        ('feature', (1,), ['feature'], 'single value'),
        ('feature', (2,), ['feature.0', 'feature.1'], 'multiple values'),
        ('feature', (['.Si', '.Mg'],), ['feature.Si', 'feature.Mg'], 'named headers'),
        ('feature', (10,), [f'feature.{i}' for i in range(10)], 'single digit values'),
        ('feature', (11,), [f'feature.{i:02d}' for i in range(11)], 'multiple digit values'),
        ('feature', (2, 2), ['feature.0.0', 'feature.0.1', 'feature.1.0', 'feature.1.1'], '2-dimensional array'),
        (
            'feature',
            (['.A', '.B'], 2),
            ['feature.A.0', 'feature.A.1', 'feature.B.0', 'feature.B.1'],
            '2-dimensional array with named headers',
        ),
        (
            'feature',
            (['.A', '.B'], ['.Si', '.Mg']),
            ['feature.A.Si', 'feature.A.Mg', 'feature.B.Si', 'feature.B.Mg'],
            '2-dimensional array with multiple named headers',
        ),
        ('feature', (1, 1, 2, 1), ['feature.0', 'feature.1'], 'multi-dimensional array with bogus dimensions'),
        (
            'feature',
            (2, 2, 2),
            [
                'feature.0.0.0',
                'feature.0.0.1',
                'feature.0.1.0',
                'feature.0.1.1',
                'feature.1.0.0',
                'feature.1.0.1',
                'feature.1.1.0',
                'feature.1.1.1',
            ],
            'multi-dimensional array with bogus dimensions',
        ),
    ],
)
def test_csv_builder_get_headers(
    prefix: str, dimensions: tuple[int | list[str]], expected_output: np.ndarray, msg: str
):
    try:
        actual_output = DataFileBuilderCsv._get_headers(prefix, dimensions)
        np.testing.assert_array_equal(actual_output, expected_output)
    except Exception as _:
        pytest.fail(msg)
