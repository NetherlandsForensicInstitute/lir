from pathlib import Path

import confidence

from lir.main import initialize_experiments


def test_parse_examples():
    examples_dir = Path(__file__).parent.parent / 'examples'
    for yaml_file in examples_dir.rglob('*.yaml'):
        try:
            initialize_experiments(confidence.loadf(yaml_file))
        except Exception as e:
            raise ValueError(f'{yaml_file}: {e}')
