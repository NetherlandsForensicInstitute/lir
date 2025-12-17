# Contributing Guide

Thank you for your interest in contributing to this project!

This guide outlines the process for setting up the project, reporting issues, proposing features, and submitting pull requests.

## Table of Contents

1. [Setting up the project locally](#setting-up-the-project-locally)
2. [Ways to contribute](#ways-to-contribute)


## Setting up the project locally

When cloning and setting up the project for the first time, please follow these steps (once).

### Source code

Clone the repository as follows:

```shell
git clone https://github.com/NetherlandsForensicInstitute/lir.git
```

This project uses [pdm](https://pdm-project.org/en/latest/) as a dependency manager. For installation of PDM, please consult the
[PDM project website](https://pdm-project.org/en/latest/#installation). It is encouraged to use PDM when contributing to this package.

Having PDM installed, install all dependencies of the project, run the following command to install the project
dependencies and `dev` dependencies used in local development.

```shell
pdm sync -G dev
```

A `.venv` directory will be created and used by PDM by default to run the python code as defined in the PDM run scripts.

This will give you the command to launch LIR with all settings in place:

```shell
pdm run lir --help
```

### Adding new dependencies
New dependencies should be installed through `pdm add <dependency_name>`.
New develop only, i.e. "dev" dependencies should be added using `pdm add <dev_dependency_name> -G dev`.

When developing locally, the following PDM scripts can be employed:
- Run linting / formatting / static analysis: `pdm check`
- Run tests: `pdm test`
- Run all checks and tests: `pdm all`

Dependencies can be upgraded using `pdm install` which also updates the `pdm.lock` lockfile.
Please only use `pdm install` if you want/need to update the package constraints and know what you're doing, otherwise
use `pdm sync` (or `pdm sync -G dev` to include the development dependencies).

### Setting up git pre-commit hook (Optional)
To run all checks before committing, you can optionally add a git pre-commit hook which ensures all checks and balances are green
before making a new commit.

Copy the `pre-commit.example` file to the `.git/hooks` folder within this project and rename it to `pre-commit`.
Next, make sure the `pre-commit` file is executable. You can run the following shell commands in the (PyCharm) terminal
from the root of the project:

```shell
cp pre-commit.example .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

---

## Ways to Contribute

You can contribute in several ways:

- Reporting bugs
- Requesting new features
- Improving documentation
- Submitting code via pull requests
- Testing or reviewing existing PRs

### 1. Issues

- Check existing issues before opening a new one
- Use clear titles and detailed descriptions
- Include steps to reproduce bugs whenever possible
- If applicable, please add appropriate labels (e.g., `bug`, `enhancement`, `documentation`)
  - There are multiple project specific labels as well, e.g. `module:plotting`

### 2. Working on New Features

- Discuss large changes in an issue before starting development
- Create a dedicated branch, e.g., `feature/your-feature-name`
- Keep your code consistent with the project’s style and structure

### 3. Pull Requests

- Make sure your fork/branch is up to date with the `main` branch
- Provide a clear explanation of what the PR does and why
- Add tests if the project includes test coverage
- Ensure all existing tests pass
- Keep PRs small and focused to make reviews easier

### 4. Code Style & Conventions

- Follow the project’s established coding standards as configured in `pyproject.toml`
  - This project recommends using `pdm` as the preferred package manager, which provides access to `pdm run all`, performing the checks automatically.
- Use clear and descriptive commit messages
- Add comments or documentation where necessary

### 5. Documentation

Please update the documentation when adding or modifying functionality. The documentation is build from the markdown files
in the `docs/` directory using [MkDocs](https://www.mkdocs.org/) and published to GitHub pages. The latest version of
the documentation resides at https://netherlandsforensicinstitute.github.io/lir/.

For each Pull Request, please include any relevant updates in the `docs/*.md` files (preferably in a separate commit). The
documentation will be rebuild upon merging the work into the `main` branch, as part of a GitHub workflow.

The documentation can be generated locally (for inspection) in multiple ways:
 - `pdm run serve` provides a preview webserver to browse through the documentation using a web browser
 - `pdm run mkdocs build` generates a `docs_html` output directory for manual inspection

Please also modify the README, wiki, or inline documentation as needed.

### 7. Releases

- Releases are handled by the maintainers
- All releases follow semantic versioning: `major.minor.patch`

### 8. Contact

- If you have any questions or suggestions, please open an issue
