import shutil
import subprocess
from pathlib import Path

import pytest

from lir.versions import _git_repo_info, collect_version_info


requires_git = pytest.mark.skipif(shutil.which('git') is None, reason='git is not available')


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
