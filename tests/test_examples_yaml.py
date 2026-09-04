import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

import confidence
import pytest

from lir.main import initialize_experiments
from lir.util import validate_yaml


EXAMPLE_DIR = Path(__file__).parent.parent / 'examples'
EXAMPLE_FILES = list(EXAMPLE_DIR.rglob('*.yaml'))


@pytest.mark.parametrize('yaml_file', EXAMPLE_FILES)
def test_parse_examples(yaml_file: Path):
    initialize_experiments(confidence.loadf(yaml_file))


@pytest.mark.parametrize('yaml_file', EXAMPLE_FILES)
def test_validate_examples(yaml_file: Path):
    validate_yaml(yaml_file)


def _check_directory_listing(listing_file: Path, output_dir: Path):
    actual_listing = sorted(_get_directory_listing(output_dir))

    # write listing if not yet available
    if not listing_file.exists():
        with open(listing_file, 'w') as f:
            f.write('\n'.join(actual_listing))

        raise ValueError(f'please check and commit directory listing at {listing_file} and re-run the test')

    # read listing
    with open(listing_file) as f:
        expected_listing = [line.strip() for line in f.readlines()]

    # check listing
    assert actual_listing == expected_listing


def _get_directory_listing(path: Path) -> Iterable[str]:
    for root, _, files in os.walk(path):
        for file in files:
            file = (Path(root) / file).relative_to(path)
            yield str(file)


@pytest.mark.parametrize('yaml_file', EXAMPLE_FILES)
def test_run_examples(yaml_file: Path):
    output_path = Path('tests/yaml_output')
    yaml_override_file = Path('tests/examples_yaml_resources') / yaml_file.name
    listing_file = yaml_override_file.with_suffix('.lst')

    run_yaml(yaml_file, yaml_override_file, listing_file, output_path)


def run_yaml(yaml_file: Path, yaml_override_file: Path | None, listing_file: Path, output_path: Path | None = None):
    """
    Check and run a YAML file.

    Parameters
    ----------
    yaml_file : Path
        The YAML file to run.
    yaml_override_file : Path, optional
        Another YAML file to override the primary YAML file.
    listing_file : Path
        A file with a directory listing that should be the result of the run.
    output_path : str, optional
        An optional output path for the results.
    """
    if output_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_yaml(yaml_file, yaml_override_file, listing_file, Path(tmpdir))
            return

    # Clean potential left-over output from running previous test
    if output_path.exists():
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True)

    configuration = confidence.Configuration(
        confidence.loadf(yaml_file),  # example YAML
        confidence.loadf(yaml_override_file) if yaml_override_file else {},  # override values
        {
            'output_path': str(output_path),
        },
    )

    experiments, _ = initialize_experiments(configuration)

    for name, experiment_definition in experiments.items():
        try:
            experiment_definition.run()
        except Exception as e:
            raise RuntimeError(f"Experiment '{name}' in '{yaml_file}' failed to run: {e}")

    _check_directory_listing(listing_file, configuration.output_path)
