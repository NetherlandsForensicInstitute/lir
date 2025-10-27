LIR Python Likelihood Ratio Toolkit
===================================

Toolkit for developing, optimising and evaluating Likelihood Ratio (LR) systems. This allows benchmarking of LR systems
on different datasets, investigating impact of different sampling schemes or techniques, and doing case-based validation
and computation of case LRs.

LIR was first released in 2020 and redesigned from scratch in 2025.

Installation
------------

The easiest way to install LIR is to use `pip`:

```shell
pip install lir
```

Terminology
-----------

- **lr system**: An algorithm to calculate likelihood ratios.
- **lr system architecture**: A way to compose an LR system, e.g. feature based, specific source, etc.
- **source**: Something that can generate instances, or where instances are derived from. This is typically the level
  relevant to the forensic question and the hypotheses. Examples**: a glass pane, a person whose face may be pictured, a
  speaker of the voice, a shoe.
- **instance**: A single manifestation of a source. Traces and reference samples are instances of a source. In a feature
  based system, instances are used as the building blocks for modeling hypotheses. Examples**: the measurements on a
  fragment of glass, a face image, a voice recording, a shoe print.
- **pair**: A combination of two groups of instances. The instance groups may be same source or different source. In a
  common-source system, pairs are used as the building blocks for modeling hypotheses. A group may contain of only one
  instance, or it may consist of multiple repeated measurements of the same source that are compared as one unit.
- **data set**: A set of instances and/or pairs, labeled or unlabeled, that may be used for calculating likelihood ratios.
- **label**: The ground-truth value for an instance or a pair. The label may be on the level of the hypothesis (e.g. H1,
  H2), or on the level of the source (e.g. Speaker1, Speaker2). Hypothesis labels may derived from source labels.
  In case of a labeled data set, the ground truth (i.e. labels) is known. This will typically be the data that is used
  for development, analysis or validation of an LR system. Unlabeled data has no ground truth. This will typically be
  the application data, or case data in a forensic setting.
- **binary data**: A data set with exactly two different labels, for specific source evaluation (e.g. H1, H2).
- **multiclass data**: A data set with an arbitrary number of labels, for common source evaluation or for reduction to
  binary data for specific source evaluation.
- **data strategy**: The way in which the data are assigned to different applications (validation, calibration, etc.) within
  the lr system. Well known strategies are train/test split, cross-validation, leave-one-out.
- **data provider**: A method for making a data set available, e.g. by reading from disk.
- **run**: Calculations for an lr system on a specific data set, as part of an experiment.
- **experiment**: A series of one or more runs to calculate lrs, to measure system performance, to evaluate the effect of
  varying system parameters, or to optimize system parameters.
- **experiment strategy**: A strategy for specifying system parameter values, e.g. single run, grid search, etc.


Usage
-----

This repository offers both a Python API and a command-line interface.


Command-line interface
----------------------

Evaluate an LR system using the command-line interface as follows:
1. define your data, LR system and experiments in a YAML file;
2. run `lir --setup <yaml file>`.

The `examples` folder may be a good starting point for setting up an experiment.

The elements of the experiment configuration YAML are looked up in the registry. The following lists all available
elements in the registry.

```commandline
pdm run benchmark --list-registry
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
- Run linting / formatting / static analysis: `pdm run check`
- Run tests: `pdm run test`
- Run all checks and tests: `pdm run all`
