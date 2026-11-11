import datetime
import importlib.metadata
import platform
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import confidence


def _git(path: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(  # noqa: S603
            ['git', '-C', str(path), *args],  # noqa: S607
            check=True,
            capture_output=True,
        )
        return result.stdout.decode('utf-8').strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_repo_info(path: Path) -> tuple[Path | None, str | None]:
    output = _git(path, 'rev-parse', '--show-toplevel', 'HEAD')
    if output is None:
        return None, None
    toplevel, commit_hash = output.splitlines()
    return Path(toplevel), commit_hash


def _extract_hash_from_version(version: str | None) -> str | None:
    if not version:
        return None
    match = re.search(r'\+g([0-9a-f]+)', version)
    return match.group(1) if match else None


def _installed_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _pyproject_info(git_root: Path) -> tuple[str, str | None]:
    name = git_root.name
    version = None
    try:
        project = tomllib.loads((git_root / 'pyproject.toml').read_text(encoding='utf-8')).get('project')
    except (OSError, tomllib.TOMLDecodeError):
        project = None
    if isinstance(project, dict):
        if isinstance(project.get('name'), str):
            name = project['name']
        if isinstance(project.get('version'), str):
            version = project['version']
    return name, version


def _package_info(package_name: str, path: Path) -> dict[str, str | None]:
    root, commit_hash = _git_repo_info(path)
    if root is not None and _pyproject_info(root)[0] != package_name:
        root, commit_hash = None, None  # package is not checked out here (e.g. installed in a venv within a repository)
    version = _installed_version(package_name)
    return {
        'version': version,
        'commit_hash': commit_hash or _extract_hash_from_version(version),
    }


def _collect_packages_info(config_yaml_path: Path | None = None) -> dict[str, dict[str, str | None]]:
    packages = {'lir': _package_info('lir', Path(__file__))}

    paths = [Path.cwd()]
    if config_yaml_path is not None:
        paths.append(config_yaml_path.resolve().parent)
    for path in paths:
        root, commit_hash = _git_repo_info(path)
        if root is not None:
            name, version = _pyproject_info(root)
            packages.setdefault(
                name,
                {
                    'version': version or _installed_version(name),
                    'commit_hash': commit_hash,
                },
            )

    return dict(sorted(packages.items()))


def collect_version_info(
    config_yaml_path: Path | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """
    Collect system metadata and version information for ``lir`` and the current project.

    Parameters
    ----------
    config_yaml_path : Path | None, optional
        The path to the YAML file describing the experiment configuration, by default None.
    command : list[str] | None, optional
        The command-line arguments used to invoke the run, by default None.

    Returns
    -------
    dict[str, Any]
        A dictionary containing ``system`` and ``packages`` metadata.
    """
    return {
        'system': {
            'timestamp': datetime.datetime.now(datetime.UTC).astimezone().isoformat(),
            'python_version': platform.python_version(),
            'os': platform.platform(),
            'architecture': platform.machine(),
            'hostname': platform.node(),
            'working_directory': str(Path.cwd()),
            'command': command if command is not None else list(sys.argv),
        },
        'packages': _collect_packages_info(config_yaml_path),
    }


def write_versions_yaml(
    output_dir: Path,
    config_yaml_path: Path | None = None,
    command: list[str] | None = None,
) -> None:
    """
    Write system and package version information to ``versions.yaml`` in the output directory.

    Parameters
    ----------
    output_dir : Path
        The directory where ``versions.yaml`` should be written.
    config_yaml_path : Path | None, optional
        The path to the YAML file describing the experiment configuration, by default None.
    command : list[str] | None, optional
        The command-line arguments used to invoke the run, by default None.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    version_info = collect_version_info(config_yaml_path, command=command)
    confidence.dumpf(version_info, output_dir / 'versions.yaml')
