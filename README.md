LIR Python Likelihood Ratio Toolkit
===================================

Toolkit for developing, optimising and evaluating Likelihood Ratio (LR) systems. This allows benchmarking of LR systems
on different datasets, investigating impact of different sampling schemes or techniques, and doing case-based validation
and computation of case LRs.

LIR was first released in 2020 and redesigned from scratch in 2025, replacing the [previous repository](https://github.com/NetherlandsForensicInstitute/lir-deprecated).

Documentation
-------------

Please consult [the dedicated documentation](https://netherlandsforensicinstitute.github.io/lir/) for a comprehensive overview of LiR, terminology and more on developing LR systems.


Installation
------------

LIR is compatible with Python 3.11 and later. The easiest way to install LIR is to use `pip`:

```shell
pip install lir
```

Usage
-----

This repository offers both a Python API and a command-line interface.


Command-line interface
----------------------

Evaluate an LR system using the command-line interface as follows:

1. define your data, LR system and experiments in a YAML file;
2. run `lir <yaml file>`.

The `examples` folder may be a good starting point for setting up an experiment.

The elements of the experiment configuration YAML are looked up in the registry. The following lists all available
elements in the registry.

```commandline
lir --list-registry
```


Datasets
--------
There are currently a number of datasets implemented for this project:

- glass: LA-ICP-MS measurements of elemental concentration from floatglass. The data will be downloaded automatically from https://github.com/NetherlandsForensicInstitute/elemental_composition_glass when used in the pipeline for the first time.

### Simulations
It is straightforward to simulate data for experimentation. Currently two very simple simulations
`synthesized_normal_binary` and `synthesized_normal_multiclass` are available, with sources and measurements drawn from
normal distributions.


Development
-----------

### Source code

Clone the repository as follows:

```shell
git clone https://github.com/NetherlandsForensicInstitute/lir.git
```

This project uses [pdm](https://pdm-project.org/en/latest/) as a dependency manager. For installation of PDM, please consult the
[PDM project website](https://pdm-project.org/en/latest/#installation).

Having PDM installed, install all dependencies of the project, run the following command to install the project
dependencies.

```shell
pdm install
```

A `.venv` directory will be created and used by PDM by default to run the python code as defined in the PDM run scripts.

This will give you the command to launch LIR with all settings in place:

```shell
pdm lir --help
```

### Setting up git pre-commit hook
To run all checks before committing, you can add a git pre-commit hook which ensures all checks and balances are green
before making a new commit.

Copy the `pre-commit.example` file to the `.git/hooks` folder within this project and rename it to `pre-commit`. 
Next, make sure the `pre-commit` file is executable. You can run the following shell commands in the (PyCharm) terminal
from the root of the project:

```shell
cp pre-commit.example .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### Adding new dependencies
New dependencies should be installed through `pdm add <dependency_name>`.

When developing locally, the following PDM scripts can be employed:
- Run linting / formatting / static analysis: `pdm check`
- Run tests: `pdm test`
- Run all checks and tests: `pdm all`
