import sys
import tempfile
from pathlib import Path

import pytest
import requests_cache

from tests.test_examples_yaml import run_yaml


SNIPPET_DIR = Path(__file__).parent.parent
SNIPPET_FILES = list(SNIPPET_DIR.rglob('*.yaml'))

sys.path.append(str(Path(__file__).parent))


@pytest.mark.parametrize('yaml_file', SNIPPET_FILES)
def test_run_examples(yaml_file: Path):
    listing_file = yaml_file.parent / 'resources' / yaml_file.with_suffix('.lst').name

    with (
        requests_cache.enabled(backend='filesystem', cache_name='tests/examples_yaml_resources/cache'),
        tempfile.TemporaryDirectory() as tmpdir,
    ):
        run_yaml(yaml_file, None, listing_file, Path(tmpdir))
