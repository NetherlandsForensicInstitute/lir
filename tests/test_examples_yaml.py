import os
import shutil

from pathlib import Path

import pytest
import confidence

from lir.main import initialize_experiments


EXAMPLE_DIR = Path(__file__).parent.parent / "examples"
EXAMPLE_FILES = list(EXAMPLE_DIR.rglob("*.yaml"))


def test_parse_examples():
    for yaml_file in EXAMPLE_FILES:
        try:
            initialize_experiments(confidence.loadf(yaml_file))
        except Exception as e:
            raise ValueError(f"{yaml_file}: {e}")


@pytest.mark.ci
def test_run_examples():
    output_path = Path('tests/yaml_output')

    # Potentially skip certain example YAML configurations
    skipped_yaml_files = (
        # 'optuna.yaml',
    )

    # Start from a clean slate
    if output_path.exists():
        shutil.rmtree(output_path)

    os.makedirs(output_path)

    for i, yaml_file in enumerate(EXAMPLE_FILES):
        if yaml_file in skipped_yaml_files:
            continue

        configuration = confidence.Configuration(
            confidence.loadf(yaml_file),
            {'output_path': f'{output_path}/{i}'},
        )

        experiments, _ = initialize_experiments(configuration)

        for name, experiment_definition in experiments.items():
            try:
                experiment_definition.run()
            except Exception as e:
                raise RuntimeError(f"Experiment '{name}' in '{yaml_file}' failed to run: {e}")
