# Agent Instructions

General instructions for AI agents working in this repository.

## Project overview

`lir` is a Python package for optimising and evaluating Likelihood Ratio (LR) systems,
developed by the Netherlands Forensics Institute. It reads experiment configuration from
YAML files, runs experiments (e.g. model selection via Optuna, inference via PyMC/MCMC),
and computes metrics, bounds, and plots.

## Tooling and setup

- Python >= 3.12, < 3.15 (CI tests on 3.12 and 3.14).
- [pdm](https://pdm-project.org/) is the dependency manager. Set up with `pdm sync -G dev`.
- Never edit `pdm.lock` by hand. Add dependencies with `pdm add` (use `-G dev` for
  dev-only dependencies), then commit the updated lock file. CI verifies it with
  `pdm lock --check`.
- if the virtual environment is broken, do not fix it, but use `pdm` to create a new one named `.venv-agent`.

## Common commands (run from the repository root)

- `pdm check` — formatting check, linting, mypy, and docstring linting
  (`ruff format --diff`, `ruff check`, `mypy lir`, `numpydoc lint`).
- `pdm test` — full test suite with coverage: `pytest tests/ docs/snippets/py/`
  (runs with `--strict-markers -W error`).
- `pdm quicktest` — run only the tests affected by local changes (pytest-testmon).
- `pdm all` — `pdm check` + `pdm test`.
- `pdm fix-all` — `ruff check --fix` + `ruff format`.
- `pdm generate-docs` — build the Sphinx documentation with `--fail-on-warning`.
- `pdm lir` — run the CLI entry point (`lir.main:main`).

Before opening a pull request, `pdm check` and `pdm test` must pass locally.
A pre-commit hook that runs these checks can be installed from `pre-commit.example`.

## Code style

- Follow the `ruff` configuration in `pyproject.toml`: line length 120, single quotes,
  two blank lines after imports, numpy docstring convention.
- `mypy` runs on `lir/` only, with strict typing: every function must be fully typed
  (`allow_untyped_defs = false`, `strict_optional = true`).
- Docstrings use the numpy style and are linted by `numpydoc`; validate with
  `pdm lint-docs`.
- Tests in `tests/` are exempt from the docstring rules (`D`) and may use `assert` (`S101`).

## Repository layout

- `lir/` — package source, organised by concern:
  - `config/` — YAML config parsing, schema generation and validation, hyperparameter substitution
  - `algorithms/` — LR-related algorithms (e.g. isotonic regression, KDE, logistic regression, MCMC)
  - `aggregation/`, `metrics/`, `bounding.py` — evaluation of LR systems (aggregation, metrics, invariance bounds)
  - `data/`, `data_setup/`, `datasets/`, `splitting_strategies/` — data handling and experiment data setup
  - `experiments/` — experiment orchestration (single runs, model selection, sensitivity analysis)
  - `lrsystems/` — LR system implementations (e.g. score-based, inference)
  - `plotting/`, `transform/`, `registry.py`, `resources/` — output, transformations, and extension registry
- `tests/` — pytest suite mirroring the package layout. `tests/golden_master/` holds
  golden-master tests; `tests/examples_yaml_resources/` holds example YAML configs used by tests.
- `docs/` — Sphinx documentation (RST). `docs/snippets/py/` contains code snippets that
  are executed as part of the test suite; keep them runnable.
- `examples/` — example YAML configurations for the CLI.
- `CONTRIBUTING.md` — contributor guide (setup, conventions, releases).

## Documentation

- The documentation is built with Sphinx from `docs/*.rst` and built with
  `--fail-on-warning` in CI. When adding or modifying functionality, update the
  relevant `docs/*.rst` files (preferably in a separate commit).

## Commits and pull requests

- Use clear, descriptive commit messages; they form the basis of the project changelog.
  Follow the existing `type: short description` style (e.g. `models: add tests for ...`).
- Keep pull requests small and focused. Add tests for new functionality and ensure all
  existing tests pass.
