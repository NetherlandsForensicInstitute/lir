import os
import shutil
from collections.abc import Iterable
from pathlib import Path

import confidence
import pytest

from lir.main import initialize_experiments


EXAMPLE_DIR = Path(__file__).parent.parent / 'examples'
EXAMPLE_FILES = list(EXAMPLE_DIR.rglob('*.yaml'))


def test_parse_examples():
    for yaml_file in EXAMPLE_FILES:
        try:
            initialize_experiments(confidence.loadf(yaml_file))
        except Exception as e:
            raise ValueError(f'{yaml_file}: {e}')


def _check_directory_listing(yaml_file: Path, output_dir: Path):
    actual_listing = sorted(_get_directory_listing(output_dir))

    listing_file = Path(__file__).parent / 'examples_yaml_resources' / f'{yaml_file.name.removesuffix(".yaml")}.lst'

    # write listing if not yet available
    if not listing_file.exists():
        with open(listing_file, 'w') as f:
            f.write('\n'.join(actual_listing))

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
    example_resources_path = Path('tests/examples_yaml_resources')

    # Clean potential left-over output from running previous test
    if output_path.exists():
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True)

    yaml_override_file = example_resources_path / yaml_file.name
    configuration = confidence.Configuration(
        confidence.loadf(yaml_file),  # example YAML
        confidence.loadf(yaml_override_file),  # override values
    )

    experiments, _ = initialize_experiments(configuration)

    for name, experiment_definition in experiments.items():
        try:
            experiment_definition.run()
        except Exception as e:
            raise RuntimeError(f"Experiment '{name}' in '{yaml_file}' failed to run: {e}")

    _check_directory_listing(yaml_file, configuration.output_path)
