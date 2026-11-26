import importlib
import shutil
import sys
from pathlib import Path

import confidence
import pytest

import lir.versions
from lir.main import load_setup, main, write_config
from lir.versions import _extract_hash_from_version, _git_repo_info

from . import resources as resources_package


requires_git = pytest.mark.skipif(shutil.which('git') is None, reason='git is not available')


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


def test_load_setup_from_file(tmp_path):
    setup = tmp_path / 'setup.yaml'
    setup.write_text('a: 1\nb: 2\n')
    assert dict(load_setup(setup)) == {'a': 1, 'b': 2}


def test_load_setup_from_directory(tmp_path):
    (tmp_path / 'part1.yaml').write_text('a: 1\n')
    (tmp_path / 'part2.yaml').write_text('b: 2\n')
    assert dict(load_setup(tmp_path)) == {'a': 1, 'b': 2}


def test_load_setup_from_directory_includes_yml_suffix(tmp_path):
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


@requires_git
def test_main_writes_versions_yaml(tmp_path: Path):
    output_dir = tmp_path / 'out'
    main([str(config), '--experiment', 'exp2', '--set', f'output_path={output_dir}'])

    versions = confidence.loadf(output_dir / 'versions.yaml')
    assert set(versions.system) >= {
        'timestamp',
        'python_version',
        'os',
        'architecture',
        'hostname',
        'working_directory',
        'command',
    }
    lir_info = versions.packages.lir
    assert lir_info.version is not None

    # the recorded commit hash must be that of the running code: from the lir repository when lir is checked out,
    # or from the hash embedded in the installed version string otherwise
    _, commit_hash = _git_repo_info(Path(lir.versions.__file__).resolve().parent)
    assert lir_info.commit_hash == (commit_hash or _extract_hash_from_version(lir_info.version))
    assert 'numpy' not in versions.packages


def test_main_disable_versions_yaml(tmp_path: Path):
    disabled_config = tmp_path / 'setup_disabled.yaml'
    disabled_config.write_text(
        f'include_versions_in_output: false\n{config.read_text(encoding="utf-8")}', encoding='utf-8'
    )

    output_dir = tmp_path / 'out'
    main([str(disabled_config), '--experiment', 'exp2', '--set', f'output_path={output_dir}'])

    assert not (output_dir / 'versions.yaml').exists()


def test_main_invalid_include_versions_in_output(tmp_path: Path):
    output_dir = tmp_path / 'out'
    with pytest.raises(ValueError):
        main(
            [
                str(config),
                '--experiment',
                'exp2',
                '--set',
                f'output_path={output_dir}',
                '--set',
                'include_versions_in_output=not_a_bool',
            ]
        )
