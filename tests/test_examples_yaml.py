import shutil
from pathlib import Path

import confidence
import pytest

from lir.main import initialize_experiments


EXAMPLE_DIR = Path(__file__).parent.parent / 'examples'
EXAMPLE_FILES = [str(f) for f in EXAMPLE_DIR.rglob('*.yaml')]


@pytest.mark.parametrize(
    'yaml_file',
    EXAMPLE_FILES,
)
def test_parse_examples(yaml_file: str):
    initialize_experiments(confidence.loadf(yaml_file))


@pytest.mark.parametrize(
    'yaml_file',
    EXAMPLE_FILES,
)
def test_run_examples(yaml_file: str):
    output_path = Path('tests/yaml_output')
    example_overrides_path = Path('tests/example_yamls_overrides')

    # Clean potential left-over output from running previous test
    if output_path.exists():
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True)

    configuration = confidence.Configuration(
        confidence.loadf(yaml_file),  # example YAML
        confidence.loadf(example_overrides_path / Path(yaml_file).name),  # override values
    )

    experiments, _ = initialize_experiments(configuration)

    for name, experiment_definition in experiments.items():
        try:
            experiment_definition.run()
        except Exception as e:
            raise RuntimeError(f"Experiment '{name}' in '{yaml_file}' failed to run: {e}")
