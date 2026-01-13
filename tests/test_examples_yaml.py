import shutil
from pathlib import Path

import confidence

from lir.main import initialize_experiments


EXAMPLE_DIR = Path(__file__).parent.parent / 'examples'
EXAMPLE_FILES = list(EXAMPLE_DIR.rglob('*.yaml'))


def test_parse_examples():
    for yaml_file in EXAMPLE_FILES:
        try:
            initialize_experiments(confidence.loadf(yaml_file))
        except Exception as e:
            raise ValueError(f'{yaml_file}: {e}')


def test_run_examples():
    output_path = Path('tests/yaml_output')
    example_overrides_path = Path('tests/example_yamls_overrides')

    # Clean potential left-over output from running previous test
    if output_path.exists():
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True)

    for yaml_file in EXAMPLE_FILES:
        configuration = confidence.Configuration(
            confidence.loadf(yaml_file),  # example YAML
            confidence.loadf(example_overrides_path / yaml_file.name),  # override values
        )

        experiments, _ = initialize_experiments(configuration)

        for name, experiment_definition in experiments.items():
            try:
                experiment_definition.run()
            except Exception as e:
                raise RuntimeError(f"Experiment '{name}' in '{yaml_file}' failed to run: {e}")
