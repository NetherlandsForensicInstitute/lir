from pathlib import Path

import confidence

from lir.main import initialize_experiments


EXAMPLE_DIR = Path(__file__).parent.parent / "examples"
EXAMPLE_FILES = list(EXAMPLE_DIR.rglob("*.yaml"))


def test_parse_examples():
    for yaml_file in EXAMPLE_FILES:
        try:
            initialize_experiments(confidence.loadf(yaml_file))
        except Exception as e:
            raise ValueError(f'{yaml_file}: {e}')
