
.. toctree::
   :hidden:

   self
   guides
   reference
   API reference <api/index>
   terminology


Getting started
===============

Toolkit for developing, optimising and evaluating Likelihood Ratio (LR) systems. This allows benchmarking of LR systems
on different datasets, investigating impact of different sampling schemes or techniques, and doing case-based validation
and computation of case LRs.

LIR was first released in 2020 and redesigned from scratch in 2025, replacing the `previous repository`_.

.. _previous repository: https://github.com/NetherlandsForensicInstitute/lir-deprecated


Installation
------------

LIR is compatible with Python 3.12 and later. The easiest way to install LIR is to use ``pip``:

.. code-block:: bash

   pip install lir


For more detailed instructions of the CLI please refer to the project `README.md`_.

.. _README.md: https://github.com/NetherlandsForensicInstitute/lir/blob/main/README.md#installation


Usage
-----

This repository offers both a Python API and a command-line interface.


Command-line interface
----------------------

To evaluate an LR system using the command-line interface, define your experiments in a YAML file and run lir:

.. code-block:: bash

    lir <yaml file>


The ``examples`` folder may be a good starting point for setting up an experiment.

The elements of the experiment configuration YAML are looked up in the registry. The following lists all available
elements in the registry.

.. code-block:: bash

    lir --list-registry


Contributing
------------
If you want to contribute to the LiR project, please follow the `CONTRIBUTING.md`_ guidelines, which include the instructions to set up LiR for local development.

.. _CONTRIBUTING.md: https://github.com/NetherlandsForensicInstitute/lir/blob/main/CONTRIBUTING.md 
