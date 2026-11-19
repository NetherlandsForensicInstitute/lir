import importlib
import shutil
import subprocess
import sys
from pathlib import Path

import confidence
import pytest

import lir.versions
from lir.main import _to_bool, load_setup, main, write_config
from lir.versions import _extract_hash_from_version, _git_repo_info, collect_version_info

from . import resources as resources_package


requires_git = pytest.mark.skipif(shutil.which('git') is None, reason='git is not available')


def raise_on_exit(exitval: int):
    raise ValueError(f'exit: {exitval}')


config = importlib.resources.files(resources_package) / 'setup.yaml'
config_dir = importlib.resources.files(resources_package) / 'setup_dir'
sys.exit = raise_on_exit


def run_git(cwd: Path, *args: str) -> str:
    result = subprocess.run(  # noqa: S603
        ['git', *args],  # noqa: S607
        cwd=cwd,
        check=True,
        capture_output=True,
    )
    return result.stdout.decode('utf-8').strip()


def create_git_repo(repo_dir: Path) -> str:
    """Initialize a git repository in the given directory and return the commit hash."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    run_git(repo_dir, 'init', '-q')
    run_git(repo_dir, 'add', '.')
    run_git(repo_dir, '-c', 'user.name=test', '-c', 'user.email=test@example.com', 'commit', '-q', '-m', 'init')
    return run_git(repo_dir, 'rev-parse', 'HEAD')


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
    output_dir = tmp_path / 'out'
    main(
        [
            str(config),
            '--experiment',
            'exp2',
            '--set',
            f'output_path={output_dir}',
            '--set',
            'include_versions_in_output=false',
        ]
    )

    assert not (output_dir / 'versions.yaml').exists()


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (True, True),
        (False, False),
        ('true', True),
        ('True', True),
        ('false', False),
        ('FALSE', False),
        (None, True),
    ],
)
def test_to_bool(value: bool | str | None, expected: bool):
    assert _to_bool(value) == expected


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


@requires_git
def test_collect_version_info_current_project(tmp_path: Path):
    repo_dir = tmp_path / 'lr_module_scratch'
    repo_dir.mkdir()
    (repo_dir / 'pyproject.toml').write_text(
        '[project]\nname = "lr_module_scratch"\nversion = "0.2.0"\n', encoding='utf-8'
    )
    setup_yaml = repo_dir / 'setup.yaml'
    setup_yaml.write_text('output_path: out\n', encoding='utf-8')
    expected_commit = create_git_repo(repo_dir)

    packages = collect_version_info(setup_yaml, command=['lir'])['packages']

    assert packages['lr_module_scratch'] == {'version': '0.2.0', 'commit_hash': expected_commit}


@requires_git
def test_git_repo_info(tmp_path: Path):
    repo_dir = tmp_path / 'repo'
    repo_dir.mkdir()
    (repo_dir / 'subdir').mkdir()
    (repo_dir / 'file.txt').write_text('content', encoding='utf-8')
    expected_commit = create_git_repo(repo_dir)

    assert _git_repo_info(repo_dir) == (repo_dir.resolve(), expected_commit)
    # resolution also works from a subdirectory of the repository
    assert _git_repo_info(repo_dir / 'subdir') == (repo_dir.resolve(), expected_commit)


def test_git_repo_info_without_git_repository(tmp_path: Path):
    assert _git_repo_info(tmp_path) == (None, None)
