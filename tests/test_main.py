import importlib
import sys

import confidence
import pytest

from lir.main import load_setup, main, write_config

from . import resources as resources_package


def raise_on_exit(exitval: int):
    raise ValueError(f'exit: {exitval}')


config = importlib.resources.files(resources_package) / 'setup.yaml'
config_dir = importlib.resources.files(resources_package) / 'setup_dir'
sys.exit = raise_on_exit


@pytest.mark.parametrize(
    'command',
    [
        '--list-registry',
        # Use single configuration file
        f'--list-experiments {config}',
        f'{config}',
        f'--experiment exp1 {config}',
        f'--experiment exp1 --experimen exp2 {config}',
        # Use configuration files from a directory
        f'--list-experiments {config_dir}',
        f'{config_dir}',
        f'--experiment exp1 {config_dir}',
        # Validate a single configuration file
        f'--validate {config}',
    ],
)
def test_main_commands(command: str):
    try:
        main(command.split())
    except Exception:
        pytest.fail('Unexpected failure while parsing valid CLI arguments')


@pytest.mark.parametrize(
    'command',
    [
        '--list-regstry',
        f'--experiment exp3 {config}',
        f'{config}x',
    ],
)
def test_invalid_main_commands(command: str):
    with pytest.raises((ValueError, FileNotFoundError)):
        main(command.split())


def test_validate_directory_setup(tmp_path):
    (tmp_path / 'part1.yaml').write_text(config.read_text())
    (tmp_path / 'part2.yaml').write_text('output_path: out\n')
    main(['--validate', str(tmp_path)])


def test_validate_invalid_directory_setup(tmp_path):
    (tmp_path / 'part1.yaml').write_text('experiments: not_a_list\n')
    with pytest.raises((ValueError, FileNotFoundError)):
        main(['--validate', str(tmp_path)])


def test_load_setup_from_file(tmp_path):
    setup = tmp_path / 'setup.yaml'
    setup.write_text('a: 1\nb: 2\n')
    assert dict(load_setup(setup)) == {'a': 1, 'b': 2}


def test_load_setup_from_directory(tmp_path):
    (tmp_path / 'part1.yaml').write_text('a: 1\n')
    (tmp_path / 'part2.yaml').write_text('b: 2\n')
    assert dict(load_setup(tmp_path)) == {'a': 1, 'b': 2}


def test_load_setup_from_directory_includes_yml(tmp_path):
    (tmp_path / 'part1.yaml').write_text('a: 1\n')
    (tmp_path / 'part2.yml').write_text('b: 2\n')
    assert dict(load_setup(tmp_path)) == {'a': 1, 'b': 2}


@pytest.mark.parametrize('names', [['readme.txt', 'data.csv'], []])
def test_load_setup_from_directory_without_yaml(tmp_path, names):
    for name in names:
        (tmp_path / name).write_text('content\n')
    with pytest.raises(FileNotFoundError, match='no YAML files found'):
        load_setup(tmp_path)


def test_write_config(tmp_path):
    cfg = confidence.Configuration({'a': 1}, {'b': 2})
    output_dir = tmp_path / 'output'
    write_config(output_dir, cfg)
    saved_cfg = confidence.loadf(output_dir / 'config.yaml')
    assert dict(saved_cfg) == {'a': 1, 'b': 2}
    assert not (output_dir / 'config').exists()


def test_write_config_from_directory_setup(tmp_path):
    setup = tmp_path / 'setup'
    setup.mkdir()
    (setup / 'part1.yaml').write_text('a: 1\n')
    (setup / 'part2.yaml').write_text('b: 2\n')
    cfg = load_setup(setup)
    output_dir = tmp_path / 'output'
    write_config(output_dir, cfg)
    assert dict(confidence.loadf(output_dir / 'config.yaml')) == {'a': 1, 'b': 2}
