YAML configuration
==================

LiR experiments are configured in YAML files. This page explains the structure of a YAML configuration, the types of
values that may be used, and the extra features that the `confidence`_ package -- on which LiR relies for parsing
YAML files -- provides. The `Setting up an experiment`_ page shows how these elements are used to define an
experiment.

.. _confidence: https://github.com/NetherlandsForensicInstitute/confidence
.. _Setting up an experiment: experiment-setup.html


Sections and lists
------------------

A YAML file consists of keys and values. A key whose value is again a set of keys and values is called a
**section**; a key whose value is an ordered collection is called a **list**.

Sections correspond to Python dictionaries, and lists correspond to Python lists. For example, in the minimal
experiment setup below, ``experiments`` is a list of experiment definitions, and each experiment definition is a
section with the subsections ``data``, ``lrsystem`` and ``output``:

.. code-block:: yaml

    output_path: ./output
    experiments:
      - name: my_experiment
        strategy: single_run
        data:
          provider:
            method: glass
          splits:
            strategy: train_test_sources
            test_size: 0.5
        lrsystem:
          architecture: score_based
          preprocessing: standard_scaler
          pairing: source_pairs
          comparing: manhattan_distance
        output:
          - method: metrics
            columns:
              - cllr
              - cllr_min


Values and types
----------------

Scalar values in YAML are converted to their Python counterparts:

- integers and floats, e.g. ``3`` and ``0.5``;
- booleans, written as ``true`` or ``false`` (converted to ``True`` and ``False`` in Python);
- strings, e.g. ``my_experiment`` (quoting is only needed when the string would otherwise be ambiguous);
- ``null``: a key without a value, or a key with the value ``null`` or ``~``, is converted to ``None`` in Python.

The ``null`` value is useful to explicitly leave an optional field empty, for example:

.. code-block:: yaml

    experiments:
      - name: architecture_evaluation
        strategy: grid
        lrsystem:               # no base LR system; it is substituted by the grid search
        lrsystem_parameters:
          ...

Comments may be added anywhere with a ``#``: everything from the ``#`` to the end of the line is ignored.


Dots in declarations
--------------------

Nested sections may also be declared with a dot in the key, as an alternative to nesting with indentation. The
following two declarations are equivalent:

.. code-block:: yaml

    data:
      provider:
        method: glass

.. code-block:: yaml

    data:
      provider.method: glass

Dotted keys are expanded into nested sections anywhere in the configuration, except for keys inside a list, where
they are kept as-is. In particular, keys of an experiment definition in the ``experiments`` list may not contain
dots, so the nested form with indentation should be used there.

Only *keys* are affected by this: dots in *values* are not special. This is why a value such as
``sklearn.impute.SimpleImputer`` in a ``method`` field, or a path such as ``comparing.steps.calibration.bandwidth``,
is left as-is.


References
----------

The ``confidence`` package supports **references**: a value enclosed in ``${...}`` refers to another part of the
configuration. References are always resolved from the root of the configuration, so any section may be referred
to from anywhere in the file. This is useful to define a section once and reuse it in multiple experiments:

.. code-block:: yaml

    data:
      provider:
        method: glass

    score_based_lrsystem:
      architecture: score_based
      ...

    experiments:
      - name: validation
        strategy: single_run
        data: ${data}
        lrsystem: ${score_based_lrsystem}

The value of a reference is the value of the referenced section itself, including its type: a reference to the
integer ``5`` evaluates to ``5``, not to the string ``'5'``.

A reference may also point to a nested section by using a dot in the reference, e.g.
``${section_name.subsection_name}``:

.. code-block:: yaml

    data:
      provider:
        method: glass

    short_name: ${data.provider.method}

A reference may be embedded in a larger value, in which case the result is always a string:

.. code-block:: yaml

    name: my_experiment
    output_path: output/${name}_run

If a reference cannot be resolved, an error is raised that mentions the referenced key. Recursive references, in
which a section refers to itself directly or indirectly, are rejected as well.


Built-in variables
------------------

LiR provides a predefined variable that may be referred to in any string value:

- ``${timestamp}``: the date and time at which the experiment was started, in the format ``YYYY-MM-DD HH-MM-SS``.

For example:

.. code-block:: yaml

    output_path: output/${timestamp}_glass_example


Overriding values on the command line
-------------------------------------

Values from the configuration file may be overridden on the command line with the ``--set`` option:

.. code-block:: shell

    lir my_setup.yaml --set output_path=/tmp/results

Keys in ``--set`` may be dotted, e.g. ``--set data.provider.method=csv``: the override is applied to the top-level
sections of the configuration file, so it affects the experiments that refer to such a section. Note that values
inside a list, such as the fields of an experiment definition in the ``experiments`` list, may not be overridden
this way, and that command-line values are read as strings: a numeric field will reject a value such as
``--set data.splits.test_size=0.3``, because the value is the string ``'0.3'``, not a number.
