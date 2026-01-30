Overview of the Python API
==========================

Data handling
-------------

In LiR, a dataset is represented as an `InstanceData`_ object, be it numeric features, scores, LLRs, or something else.

The hierarchy of data models is as follows.

.. code-block::

    `InstanceData`_
    \- `FeatureData`_
    |- `PairedFeatureData`_
       \- `LLRData`_

These objects can be instantiated manually, but in an experimental setup, they are generally provided by a `DataProvider`_.
A ``DataProvider`` is a class that is specialized in generating, parsing or fetching a particular type of data.

A data provider sub class implements the ``get_instances()`` method.

Example:

.. jupyter-execute::

    import numpy as np
    from lir.data.datasets.glass import GlassData

    data_provider = GlassData(cache_dir='cache')
    glass_data = data_provider.get_instances()

    n_sources = len(np.unique(glass_data.source_ids))
    print(f'the dataset has {len(glass_data)} instances, which are measurements on {n_sources} glass fragments')


How the data are used in an experiment is determined by the `data strategy`_. This can be a simple train/test split, or a more
advanced configuration such as cross-validation.

A data strategy inherits from `DataStrategy`_ and implements an ``apply()`` method that returns an iterator of pairs of training and test sets.

Example:

.. jupyter-execute::

    from lir.data.data_strategies import MulticlassTrainTestSplit

    strategy = MulticlassTrainTestSplit(test_size=0.5)
    for training_data, test_data in strategy.apply(glass_data):
        print(f'we have {len(training_data)} instances available for training our models')
        print(f'we have {len(test_data)} instances available as test data')

.. _data strategy: api/lir.data.html#module-lir.data.data_strategies
.. _InstanceData: api/lir.data.html#lir.data.models.InstanceData
.. _FeatureData: api/lir.data.html#FeatureData
.. _PairedFeatureData: api/lir.data.html#PairedFeatureData
.. _LLRData: api/lir.data.html#LLRData
.. _DataProvider: api/lir.data.html#lir.data.models.DataProvider
.. _DataStrategy: api/lir.data.html#lir.data.models.DataStrategy
.. _Pipeline: api/lir.transform.html#lir.transform.pipeline.Pipeline


LR systems
----------

There can be many different `LR system architectures`_. Refer to the `selection guide`_ if unsure which one to use.

The exact parameters depend on the type of LR system. The main ingredient is typically a pipeline of *modules*.
The modules are executed one by one, and each module takes the output of the previous module, transforms the data,
and passes the data to the next module. The modules may reduce or expand the number of features, but never change
the number of instances.

The `Pipeline`_ class accepts any LiR module scikit-learn transformer, scikit-learn estimator, or even other
pipelines as its modules, as long as the module can work with the data.

For example, the ``StandardScaler`` transforms the data to have mean = 0 and standard deviation = 1.

Example:

.. jupyter-execute::

    from sklearn.preprocessing import StandardScaler
    from lir.transform.pipeline import Pipeline

    preprocessing_pipeline = Pipeline([
        ('scale', StandardScaler()),
    ])

    normalized_glass_data = preprocessing_pipeline.fit_apply(glass_data)


Most LR systems compare two instances and deterime whether they are from the same source or not.
We already acquired measurements on single glass fragments.
To get from single fragments to comparisons, we also need a `pairing method`.

Example:

.. jupyter-execute::

    from lir.transform.pairing import InstancePairing

    pairing_method = InstancePairing(ratio_limit=1)
    paired_glass_data = pairing_method.pair(normalized_glass_data)

    print(f'we have {len(glass_data)} measurements on glass fragments')
    print(f'from those data we made {len(paired_glass_data)} pairs')


A simple LR system could take the manhattan distance of each pair, apply some kind of calibration.

.. jupyter-execute::

    from lir.transform.distance import ManhattanDistance
    from lir.algorithms.logistic_regression import LogitCalibrator

    comparing_pipeline = Pipeline([
        ('distance', ManhattanDistance()),
        ('logit', LogitCalibrator()),
    ])

    llrs = comparing_pipeline.fit_apply(paired_glass_data)


We can use these components in a score-based LR system:

.. jupyter-execute::

    from lir.lrsystems.score_based import ScoreBasedSystem

    lrsystem = ScoreBasedSystem(preprocessing_pipeline=preprocessing_pipeline, pairing_function=pairing_method, evaluation_pipeline=comparing_pipeline)
    llrs = lrsystem.apply(glass_data)

    # plot results
    import lir.plotting
    with lir.plotting.show() as ax:
        ax.lr_histogram(llrs)

    # zoom in on the LLRs around 0
    with lir.plotting.show() as ax:
        ax.xlim(-4, 4)
        ax.lr_histogram(llrs, bins=100)


.. _LR system architectures: reference.html#lrsystem-architecture
.. _selection guide: reference.html#lrsystem_yaml
.. _pairing methods: reference.html#pairing-methods
