Extending LiR by creating a custom component
============================================

LiR is designed to be easily extensible, allowing developers to add new features and functionality without modifying
the core code. This document outlines the process for creating a custom component that can be used in an experiment
setup configuration file.

In this example, we will create a custom module that calculates the cosine similarity between two vectors.
The module will be integrated into the LiR framework and can be used in experiments just like any other built-in
component.

This example shows how to create a custom module, but the same principles apply to other types of components, such as
experiment types, LR system architectures, metrics, etc.
 

Step 1: setup a new project
---------------------------

Let's start by setting up a new project for an experiment with a simple LR system.
The following example shows a configuration file for an experiment:

.. literalinclude:: snippets/minimal-single-run.yaml
    :language: yaml

It uses glass data in CSV format, fetched from a URL.
The LR system uses the score-based architecture, :class:`~lir.lrsystems.ScoreBasedSystem`.
Manhattan distance is used to calculate the distance between pairs, and discriminate between same-source and
different-source pairs.

To run this experiment, save this snippet as ``config.yaml`` and run it with the following command:

.. code-block:: bash

    lir run config.yaml

Step 2: implement cosine similarity
-----------------------------------

Our goal is to replace manhattan distance by cosine similarity. The first thing we need to do is to implement cosine
similarity. We create a file named ``cosine_similarity.py`` and implement the cosine similarity class.

.. literalinclude:: snippets/py/cosine_similarity.py
    :language: python
    :lines: 1-6,9-11,15-27,31-33

The class we created is a child of :class:`~lir.Transformer`. This tells LiR that it is a module that transforms data.
It has an ``apply()`` method that takes pairs of feature vectors and returns a similarity score.
The cosine similarity is calculated using the dot product of the two feature vectors, divided by the product of their
norms.

This module is now ready to run in an LR system, and can be referenced in a configuration file as ``cosine_similarity.CosineSimilarity``.

.. literalinclude:: snippets/minimal-single-run-with-cosim.yaml
    :language: yaml
    :lines: 1-40,42-
    :emphasize-lines: 39-40

Try running the above example in LiR. If it does not work, make sure the Python interpreter can access the
``cosine_similarity.py`` file. It should work if the file is in the working directory or if it is in the
``PYTHONPATH``.


Step 3: add parameters
----------------------

Our cosine similarity component works without any parameters. For the sake of it, we will add a parameter to control
whether the cosine similarity should be squared or not.
We extend the ``CosineSimilarity`` class by adding an  ``__init__`` method that takes a boolean parameter:

.. literalinclude:: snippets/py/cosine_similarity.py
    :language: python
    :lines: 1-6,9-33
    :emphasize-lines: 10-11,26-27

Any key/value-pair in the configuration section of ``scorer`` (except for ``method``) will be passed to the component when it is instantiated.

.. literalinclude:: snippets/minimal-single-run-with-cosim.yaml
    :language: yaml
    :emphasize-lines: 41


Step 4: create a configuration parser (optional)
------------------------------------------------

There is a good chance that this is all you need to do to create a new component. However, if you want more control over how the component is instantiated, you can create a custom configuration parser.

We create a configuration parser by defining a function that parses a configuration section and returns a ``CosineSimilarity`` object.
We take care that the function:

- takes two arguments: a :class:`~lir.config.ConfigValue`, the configuration section in the YAML that is needed to initialize the module, and an output directory as a :class:`~pathlib.Path` object;
- returns an instance of the component (in our case, ``CosineSimilarity``);
- is marked as a configuration parser with the :deco:`~lir.config.config_parser` decorator.

.. literalinclude:: snippets/py/cosine_similarity.py
    :language: python
    :emphasize-lines: 7,36-


The ``config`` parameter contains the relevant section in the configuration YAML.
Each :class:`~lir.config.ConfigValue` object has a ``context`` attribute, which is the full path of the configuration value in the original YAML file, and a ``value`` attribute, which is the actual value.
If ``value`` is a container object such as a list or a dictionary, it is also wrapped inside a :class:`~lir.config.ConfigValue` object.
Please refer to the API documentation for more information on dealing with :class:`~lir.config.ConfigValue` objects.

.. code-block:: bash

    lir run minimal-single-run-with-cosim-parser.yaml

.. jupyter-execute::
    :hide-code:

    import tempfile
    import lir.main
    import sys

    sys.path.append('docs/snippets/py')

    with tempfile.TemporaryDirectory() as tmpdirname:
        lir.main.main(['docs/snippets/minimal-single-run-with-cosim-parser.yaml', '--set', f'output_path={tmpdirname}'])


In our case, the ``config`` parameter holds the configuration section under ``score`` (line 31), which is a dictionary of two keys, namely ``method`` and ``square``.
Since the ``method`` is already consumed to identify the configuration parser, this leaves a dictionary with just ``square`` as its single key.

The ``output_dir`` value is the path to the directory where the component may save its results, if applicable.

To use this configuration parser, simply replace ``CosineSimilarity`` in the configuration file in `Step 3: add parameters`_ with ``parse_cosine_similarity_config``.


Step 5: register your component (optional)
------------------------------------------

Registring a component creates an alias. When registered, the component can be referenced by its alias (e.g.
``cosim``) instead of its full module path (i.e. ``cosine_similarity.CosineSimilarity`` or
``cosine_similarity.parse_cosine_similarity``).

To register a component, create a file named ``registry.yaml`` in the current directory.
See the `documentation`_ of ``confidence.load_name`` for the list of locations where registry files are searched.

The cosine similarity module can be registered by adding the following entry to the ``registry.yaml`` file:

.. code-block:: yaml

    modules:
        cosine_similarity: cosine_similarity.CosineSimilarity

Or, if you want to use the configuration parser:

.. code-block:: yaml

    modules:
        cosine_similarity: cosine_similarity.parse_cosine_similarity_config

You can also use this mechanism to override specific built-in registry entries, since user-defined registry entries take precedence over built-in registry entries.

.. _documentation: https://github.com/NetherlandsForensicInstitute/confidence


Examples for other components
=============================

Other components can be added in a similar way, by creating a function with the :deco:`~lir.config.config_parser` decorator, and returning the component instance.


A data provider to read from a database
---------------------------------------

We will create a data provider that obtains its data from a Sqlite database.

Before writing the component, we create and populate a database with dummy values.

.. jupyter-execute::
    :hide-code:

    from pathlib import Path

    Path('example.db').unlink(missing_ok=True)

.. jupyter-execute::

    import sqlite3

    with sqlite3.connect('example.db') as db:
        db.execute('CREATE TABLE feature_table (hypothesis INT, feature1 FLOAT, feature2 FLOAT)')
        db.execute('INSERT INTO feature_table VALUES(1, 1.0, 1.1), (1, 1.1, 1.2), (0, 3.1, 3.3), (0, 3.0, 4.0)')


Now that we have the database, we create a data provider component to read from it.
The recommended way to do it, is to define a function that returns the data, and decorate it with :deco:`~lir.config.data.data_provider`:

.. literalinclude:: snippets/py/sqlite_reader.py
    :language: python


Alternatively, we may define a :class:`~lir.DataProvider` class. If its ``__init__`` method requires no parameters or atomic parameters only, like here, it can be used directly.
Otherwise, we'll have to use :deco:`~lir.config.config_parser`:

.. literalinclude:: snippets/py/sqlite_reader_alt.py
    :language: python


An experiment type to count the number of records in the dataset
----------------------------------------------------------------

This example shows how to create a new type of experiment. All it does is to count the number of instances in a dataset.

.. literalinclude:: snippets/py/data_counter_experiment.py
    :language: python


An output method to calculate the average LLR
---------------------------------------------

The ``output`` section of the experiments list methods to aggregate results.
This is done by :class:`~lir.aggregation.Aggregation` objects, which receive the results of each LR system run one by one.

In this example, we will create an output method that calculates the average LLR for each run, and writes it to a file.
Finally, we will calculate the average of all runs.

.. literalinclude:: snippets/py/average_llr_aggregation.py
    :language: python


A metric to calculate the average LLR
-------------------------------------

A metric is a function that takes :class:`~lir.LLRData` and returns a ``float``.
In this example, we implement the average of LLRs as a metric.

.. literalinclude:: snippets/py/average_llr_metric.py
    :language: python


A slightly more complicated example is the calculation of a weighted CLLR value, which involves the :deco:`~lir.config.config_parser` decorator.

.. literalinclude:: snippets/py/average_llr_metric_weighted.py
    :language: python


A test-only data strategy
-------------------------

Some models do not require a training phase.
In this example, we implement a data strategy that assumes we work with such models only, and offers the full dataset for testing.

.. literalinclude:: snippets/py/skip_training.py
    :language: python
