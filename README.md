LIR Python Likelihood Ratio Toolkit
===================================

Toolkit for developing, optimising and evaluating Likelihood Ratio (LR) systems. This allows benchmarking of LR systems
on different datasets, investigating impact of different sampling schemes or techniques, and doing case-based validation
and computation of case LRs.

LIR was first released in 2020 and redesigned from scratch in 2025, replacing the [previous repository](https://github.com/NetherlandsForensicInstitute/lir-deprecated).

References
-------------

- [LiR documentation](https://netherlandsforensicinstitute.github.io/lir/): comprehensive overview, terminology and more on developing LR systems
- Practitioner Guide ([branch](https://github.com/NetherlandsForensicInstitute/lir/tree/practitioner_guide) | [paper](https://doi.org/10.1016/j.forsciint.2024.111994) | [notebook](https://colab.research.google.com/github/NetherlandsForensicInstitute/lir/blob/practitioner_guide/practitioners_guide_glass.ipynb)): case study using LiR to develop an LR system using LiR
- [Quick Start](https://netherlandsforensicinstitute.github.io/lir/lrsystem_yaml/): selecting / designing the proper LR system based on your data


Installation
------------

LIR is compatible with Python 3.12 / 3.13/ 3.14. The easiest way to install LIR is to use `pip`:

```shell
pip install lir
```

Usage
-----

This repository offers both a Python API and a command-line interface.


Command-line interface
----------------------

LiR can be launched from the command line as follows:

```sh
lir --help
```

Or, alternatively:

```sh
python -m lir
```

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


Contributing / Development
-----------

Contributions are highly welcomed. If you'd like to contribute to the LiR package, please follow the steps as described
in the [CONTRIBUTING.md](CONTRIBUTING.md) file.
