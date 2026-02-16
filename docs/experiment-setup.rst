Setting up an experiment
========================

This page shows how write an experiment setup using LiR.

Before you begin, make sure you have a working `lir command`_

Experiments are defined in an experiment setup file. For a quick start, use one of the `examples`_.

An experiment setup file is a YAML file with at least the ``experiments`` property that lists the experiments, and the
``output_path`` property:

.. code-block:: yaml

    output_path: ./output  # this is where generated output is saved
    experiments:
      - ... definition of experiment 1 ...
      - ... definition of experiment 2 ...

LiR uses the ``confidence`` package for parsing YAML files. Refer to its `documentation`_ to make the most of it! In
particular, you may use the ``timestamp`` variable to substitute the current date and time in the YAML configuration,
for example:

.. code-block:: yaml

    output_path: ./${timestamp}_output  # this is where generated output is saved
    experiments:
      - ... definition of experiment 1 ...
      - ... definition of experiment 2 ...

.. _lir command: index.html
.. _examples: lrsystem_yaml.html
.. _documentation: https://github.com/NetherlandsForensicInstitute/confidence


Experiment definition
---------------------

An experiment definition has at least a ``strategy`` property. There can be as many experiment strategies as there are
reasons to setup an experiment, including:

- calculate LRs, metrics or plots for a particular LR system and dataset (e.g. the ``single_run`` strategy);
- optimize hyperparameters of an LR system for a particular dataset (e.g. the ``optuna`` strategy);
- assess the quality of a range of LR systems for model selection (e.g. the ``grid`` strategy);
- assess the sensitivity of an LR system for particular data parameters (e.g. the ``grid`` strategy);
- calculate LRs for data with unknown ground-truth.

A fully working example that uses the ``single_run`` strategy:

.. literalinclude:: snippets/minimal-single-run.yaml
    :language: yaml

This experiment definition has the following sections:

- ``name``, an arbitrary name of the experiment;
- ``strategy``, the experiment strategy;
- ``lr_system``, which defines the LR system;
- ``data``, which defines the dataset (``provider``) and how it is split into training and test data (``splits``);
- ``output``, which specifies the required output.

Some `experiment strategies`_ take additional parameters.

.. _experiment strategies: reference.html#experiment-strategies


Data organization
-----------------

The `data provider`_ delivers the dataset. It has at least the ``method`` property, and any other property is passed
as a parameter of the data provision method.

In evaluative settings, the data provider delivers:

- data on instances with hypothesis labels, e.g. for calculating specific-source LRs;
- data on instances with source ids, e.g. for calculating common-source LRs;
- data on instance pairs with hypothesis labels, e.g. for calculating common-source LRs.

The `data splitting strategy`_ defines how the data is split into a training set and a test set. It has at least the
``strategy`` property, and any other property is passed as a parameter of the data strategy. The data strategy should
be compatible with the type of data in the dataset.

Some data strategies, such as cross-validation, split the data several times. In that case, the LR system is trained
and applied for each train/test split separately, but then the test results are concatenated.

.. _data provider: reference.html#data-providers
.. _data splitting strategy: reference.html#data-strategies


LR systems
----------

The LR system section defines the LR system. There are various `architectures`_, and the architecture as well as the
processing modules should support the input data.

Depending on the architecture, there may be parameters that specify how data are processed or how instances are paired.
For example, the ``score_based`` architecture has a preprocessing method, a pairing method, and a comparing method.

A `pairing method`_ governs how instances are combined into pairs. Both the preprocessing and the comparing methods are
`modules`_ that transform the data, but *never* change the number of instances or the order of the instances. The
``pipeline`` module can be used to arrange a sequence of modules.

.. _architectures: reference.html#lr-system-architecture
.. _modules: reference.html#lr-system-modules
.. _pairing method: reference.html#pairing-methods


Output
------

Results are written to the directory in ``output_path``.

By default, a detailed log file is written to ``output_path/log.txt``.

Within ``output_path``, a directory is created with the same name as the experiment. In the above example,
the experiment only has a single run, but other experiment strategies may have multiple. In that case, a separate
directory is created for each run, with the following files:

- ``data.yaml``: the data organization that was used for the run;
- ``lrsystem.yaml``: the LR system setup that was used for the run.

Additionally, we expect to find the output specified in the ``output`` section. After running the example, the full
directory listing is as follows.

.. code-block::

    output/log.txt
    output/my first experiment/PAV.png
    output/my first experiment/data.yaml
    output/my first experiment/lrsystem.yaml
    output/my first experiment/metrics.csv


Hyperparameters and data parameters
-----------------------------------

`Experiments`_ that involve multiple runs have a ``hyperparameters`` or a ``dataparameters`` section.

.. _Experiments: reference.html#experiment-strategies

For example, the ``grid`` strategy runs the LR system for each combination of hyperparameter values.
The ``optuna`` strategy runs the LR system a fixed number of times, while trying to optimize parameter values.


Model selection
^^^^^^^^^^^^^^^

In the ``single_run`` example, we have a fully working LR system that uses logistic regression to calculate LRs. Let's
say we want to try other LR calculation methods as well, and compare the results. To make this work, we use the same
"baseline LR system", but make two modifications to the experiment:

- replace the ``single_run`` strategy by the ``grid`` strategy; and
- define a hyperparameter for the LR calculation method.

.. literalinclude:: snippets/model-selection.yaml
    :language: yaml
    :emphasize-lines: 4,26-31

This will run the LR system three times, once for each LR calculation method. All metrics are collected in
``metrics.csv`` and the output directory lists the following files.

.. code-block::

    output/log.txt
    output/my model selection/comparing.steps.calibration=isotonic/PAV.png
    output/my model selection/comparing.steps.calibration=isotonic/data.yaml
    output/my model selection/comparing.steps.calibration=isotonic/lrsystem.yaml
    output/my model selection/comparing.steps.calibration=logit/PAV.png
    output/my model selection/comparing.steps.calibration=logit/data.yaml
    output/my model selection/comparing.steps.calibration=logit/lrsystem.yaml
    output/my model selection/comparing.steps.calibration=kde/PAV.png
    output/my model selection/comparing.steps.calibration=kde/data.yaml
    output/my model selection/comparing.steps.calibration=kde/lrsystem.yaml
    output/my model selection/metrics.csv


Sensitivity analysis
^^^^^^^^^^^^^^^^^^^^

We may also want to know how wel the LR system is able to cope with few data points. This is a similar setup, but since
we vary the input data, we use ``dataparameters`` instead of ``hyperparameters``, like so:

.. literalinclude:: snippets/sensitivity-analysis.yaml
    :language: yaml
    :emphasize-lines: 21,31-39

Again, this generates the results for the different data sizes, and metrics are collected in ``metrics.csv``.

.. code-block::

    output/log.txt
    output/my sensitivity analysis/provider.head=100/PAV.png
    output/my sensitivity analysis/provider.head=100/data.yaml
    output/my sensitivity analysis/provider.head=100/lrsystem.yaml
    output/my sensitivity analysis/provider.head=200/PAV.png
    output/my sensitivity analysis/provider.head=200/data.yaml
    output/my sensitivity analysis/provider.head=200/lrsystem.yaml
    output/my sensitivity analysis/provider.head=300/PAV.png
    output/my sensitivity analysis/provider.head=300/data.yaml
    output/my sensitivity analysis/provider.head=300/lrsystem.yaml
    output/my sensitivity analysis/provider.head=400/PAV.png
    output/my sensitivity analysis/provider.head=400/data.yaml
    output/my sensitivity analysis/provider.head=400/lrsystem.yaml
    output/my sensitivity analysis/provider.head=500/PAV.png
    output/my sensitivity analysis/provider.head=500/data.yaml
    output/my sensitivity analysis/provider.head=500/lrsystem.yaml
    output/my sensitivity analysis/provider.head=600/PAV.png
    output/my sensitivity analysis/provider.head=600/data.yaml
    output/my sensitivity analysis/provider.head=600/lrsystem.yaml
    output/my sensitivity analysis/metrics.csv


Advanced use of parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^

So far, we only used categorical parameters. There are `other types of parameters`_, such as numerical parameters,
cluster parameters or constants.

In the sensitivity analysis, the data size (``provider.head``) is implicitly understood to be a categorical parameter.
We may instead use a numerical variable, which yields the exact same results:

.. literalinclude:: snippets/sensitivity-analysis-numerical.yaml
    :language: yaml
    :emphasize-lines: 32-35

In the examples above, we used the implicit form to specify the parameter values:

.. code-block:: yaml

    hyperparameters:
      - path: comparing.steps.calibration
        options:
          - logit
          - isotonic
          - kde

This is implicitly understood to be a categorical parameter. However, we can also write it in the explicit form. The
following is equivalent to the above.

.. code-block:: yaml
    :emphasize-lines: 2

    hyperparameters:
      - type: categorical
        path: comparing.steps.calibration
        options:
          - logit
          - isotonic
          - kde

In most cases, the ``type`` property can be omitted, but it may be necessary when using custom parameter types.

.. _other types of parameters: reference.html#hyperparameters
