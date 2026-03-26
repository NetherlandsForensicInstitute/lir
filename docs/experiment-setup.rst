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

An experiment definition has all the configuration required to run an experiment. There are several ways to setup an
experiment, but the most simple strategy is to use a single LR system and run it. This is the ``single_run`` strategy.
Below is a fully working example:

.. literalinclude:: snippets/minimal-single-run.yaml
    :language: yaml

Every experiment configuration has a ``strategy`` property, which defines the type of experiment, and also which
configuration settings are required. Our ``single_run`` setup has three main sections:

- ``data``, which defines the dataset (``provider``) and how it is split into training and test data (``splits``);
- ``lr_system``, which defines the LR system; and
- ``output``, which specifies the required output.

More advanced `strategies`_, such as the ``grid`` strategy and the ``optuna`` strategy, are extensions of this. They
can evaluate a range of LR systems for model selection, sensitivity analyses, or hyperparameter optimization.

.. _strategies: reference.html#experiment-strategies

For now, we'll stick with the ``single_run`` strategy. Save this setup to a file named ``minimal-single-run.yaml`` and
run it as:

.. code-block:: shell

    lir minimal-single-run.yaml

This will:

- load data from a CSV file;
- split the data randomly into a training set and a test set;
- train the LR system on the training set;
- apply the LR system on the test set;
- calculate the CLLR and the CLLR_min on the test set.

Results are written to the directory in ``output_path``.

By default, a detailed log file is written to ``output_path/log.txt``.

Within ``output_path``, a directory is created with the same name as the experiment. In the above example,
the experiment only has a single run, but other experiment strategies may have multiple. In that case, a separate
directory is created for each run, with the following files:

- ``data.yaml``: the data organization that was used for the run;
- ``lrsystem.yaml``: the LR system setup that was used for the run.

Additionally, we expect to find the output specified in the ``output`` section. After running the example, the full
directory listing is as follows.

.. jupyter-execute::
    :hide-code:

    import tempfile
    import glob
    import os
    import lir.main

    def run_and_list_directory(setup_file: str):
        with tempfile.TemporaryDirectory() as tmpdirname:
            lir.main.main([setup_file, '--set', f'output_path={tmpdirname}'])
            for filename in glob.glob(f'{tmpdirname}/**', recursive=True):
                shortname = filename[len(tmpdirname)+1:]
                if not shortname:
                    pass  # do not print the root
                elif os.path.isdir(filename):
                    print(f'{shortname}/')
                else:
                    print(shortname)

    run_and_list_directory('docs/snippets/minimal-single-run.yaml')


Data organization
-----------------

The `data provider`_ delivers the dataset. It has at least the ``method`` property, and any other property is passed
as a parameter of the data provision method.

In evaluative settings, the data provider delivers **labeled data**. In the setup above, that means that the instances
include **source ids**. We don't have hypothesis labels yet, because we evaluate the instances only after pairing them.

We could substitute the glass data for another data provider. Let's try synthesized data:

.. code-block:: yaml

    provider:
      method: synthesized_normal_multiclass  # for a comparative evaluation, we use the multiclass variant
      seed: 0                    # the random seed
      population:
        size: 100                # the number of sources
        instances_per_source: 2  # the number of instances that are drawn for each source
      dimensions:
        - mean: 0                # the average true value of all sources
          std: 1                 # the standard deviation of the true value of all sources
          error_std:             # the standard deviation of the measurement error

In the ``single_run`` example above, we used a train test split (``train_test_sources``) to split the data into a
training set and a test set. This is fine if you have plenty of data, but other `splitting strategies`_ use the data more
efficiently. We could instead use `cross validation`_:

.. _cross validation: https://en.wikipedia.org/wiki/Cross-validation_(statistics)

.. code-block:: yaml

    splits:
      strategy: cross_validation_sources
      folds: 5

Now, if we run the experiment again, the system will be trained and applied five times, on different subsets. The five
test sets are merged and we can calculate the metrics on the full dataset!

In some cases, we want full control over which instances are used for training and which ones end up in the test set.
In that case, the train/test roles are assigned by the data provider, and we can use the predefined splitting strategies
``predefined_train_test`` or ``predefined_cross_validation``.

.. code-block:: yaml

    data:
      provider:
        method: glass
        cache_dir: .glass-data
      splits:
        strategy: predefined_train_test

.. _data provider: reference.html#data-providers
.. _splitting strategies: reference.html#data-strategies

So far, we assumed that the instances are paired by the LR system. The data provider delivers source ids, but not
hypothesis labels. If the instances are not paired, and the data provider delivers hypothesis labels, we also need to
choose our splitting strategy differently. Applicable data strategies are ``train_test`` and ``cross_validation``. See
the setup file ``specific_source_evaluation.yaml`` in the ``examples`` folder for a fully working example.


LR systems
----------

The LR system section defines the LR system. There are various `architectures`_, and the architecture as well as the
processing modules should support the input data.

In the experiment setup example above, we used the ``score_based`` architecture. That means that we can specify a
preprocessing method, a pairing method, and a comparing method.

Preprocessing is done on the individual instances. The `pairing method`_ governs how instances are combined into pairs.
Comparing is done after pairing and should involve calculating LLRs.

Both the preprocessing and the comparing methods are `modules`_ that transform the data, but *never* change the number
of instances or the order of the instances. The ``pipeline`` module can be used to arrange a sequence of modules.

.. _architectures: reference.html#lr-system-architecture
.. _modules: reference.html#lr-system-modules
.. _pairing method: reference.html#pairing-methods

Above, our preprocessing was done by the scaler. The **shortest** way to write this in the setup is as follows.

.. code-block:: yaml

    lr_system:
      architecture: score_based
      preprocessing: standard_scaler
      ...

However, if we need to pass additional **parameters** to the scaler, we use the standard form.

.. code-block:: yaml

    lr_system:
      architecture: score_based
      preprocessing:
        method: standard_scaler
        with_mean: True
        with_std: True
      ...

In this form, the ``method`` specifies the module, and any other fields are passed as parameters to the module on
initialization.

Still, we may not be satisfied, because we want to use more than one module for preprocessing. So, we create a pipeline.

.. code-block:: yaml

    lr_system:
      architecture: score_based
      preprocessing:
        method: pipeline
        steps:
          imputer: sklearn.impute.SimpleImputer
          scaler:
            method: standard_scaler
            with_mean: True
            with_std: True
      ...

Output
------

The output section declares how to aggregate the results from the test set.


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
    :emphasize-lines: 3-4,41-47

This will run the LR system three times, once for each LR calculation method. All metrics are collected in
``metrics.csv`` and its contents is the following.

.. jupyter-execute::
    :hide-code:

    with tempfile.TemporaryDirectory() as tmpdirname:
        lir.main.main(['docs/snippets/model-selection.yaml', '--set', f'output_path={tmpdirname}'])
        with open(f'{tmpdirname}/my_model_selection_exp/metrics.csv', 'r') as f:
            print(f.read())



Sensitivity analysis
^^^^^^^^^^^^^^^^^^^^

We may also want to know how wel the LR system is able to cope with few data points.

Therefore, the CSV reader has the argument ``head`` to read only the first ``n`` instances. We are going to use that
argument to vary the amount of input data. This is similar to the model selection setup, but since
we vary the input data, we use ``dataparameters`` instead of ``hyperparameters``, like so:

.. literalinclude:: snippets/sensitivity-analysis.yaml
    :language: yaml
    :emphasize-lines: 3,16,42-50

Again, this generates the results for the different data sizes, and metrics are collected in ``metrics.csv``.

.. jupyter-execute::
    :hide-code:

    with tempfile.TemporaryDirectory() as tmpdirname:
        lir.main.main(['docs/snippets/sensitivity-analysis.yaml', '--set', f'output_path={tmpdirname}'])
        with open(f'{tmpdirname}/my_sensitivity_exp/metrics.csv', 'r') as f:
            print(f.read())


Advanced use of parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^

So far, we only used categorical parameters. There are `other types of parameters`_, such as numerical parameters,
cluster parameters or constants.

In the sensitivity analysis, the data size (``provider.head``) is implicitly understood to be a categorical parameter.
We may instead use a numerical variable, which yields the exact same results:

.. literalinclude:: snippets/sensitivity-analysis-numerical.yaml
    :language: yaml
    :emphasize-lines: 43-46

In the examples above, the parameters are automatically recognized to be categorical or numerical. However, we can also
explicitly specify the parameter type. The following is equivalent to the above.

.. code-block:: yaml
    :emphasize-lines: 2

    hyperparameters:
      - path: comparing.steps.to_llr
        type: categorical
        options:
          - logistic_calibrator
          - method: kde
            bandwidth: silverman
          - isotonic_calibrator

In most cases, the ``type`` property can be omitted, but it may be necessary when using custom parameter types.

.. _other types of parameters: reference.html#hyperparameters
