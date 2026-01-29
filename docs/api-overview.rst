Overview of the Python API
==========================

Data classes
------------

In LiR, a dataset is represented as an `InstanceData`_ object, be it numeric features, scores, LLRs, or something else.

Specialized sub classes are:

- `FeatureData`_, for instances which has numerical features (sub class of ``InstanceData``);
- `PairedFeatureData`_, for pairs of instances that have numerical features (sub class of ``FeatureData``);
- `LLRData`_, for LLRs, with or without intervals (sub class of ``FeatureData``).

.. _InstanceData: api/lir.data.html#lir.data.models.InstanceData
.. _FeatureData: api/lir.data.html#FeatureData
.. _PairedFeatureData: api/lir.data.html#PairedFeatureData
.. _LLRData: api/lir.data.html#LLRData

These objects can be instantiated manually, but in an experimental setup, they are generally provided by a `DataProvider`_.
A ``DataProvider`` is a class that is specialized in generating, parsing or fetching a particular type of data.

A data provider sub class implements the ``get_instances()`` method.

Example for glass data:

.. jupyter-execute::

    import numpy as np
    from lir.data.datasets.glass import GlassData
    from lir.transform.distance import ManhattanDistance
    from lir.algorithms.logistic_regression import LogitCalibrator

    # retrieve the data
    data_provider = GlassData(cache_dir='cache')
    glass_data = data_provider.get_instances()

    print(f'The glass dataset has numeric features, so it is of type {type(glass_data)}.')
    print(f'It has {len(glass_data)} instances, each of which is a measurement on a range of chemical elements.')
    print(f'The measurements are of {len(np.unique(glass_data.source_ids))} different glass fragments.')


Most LR systems compare two instances, one from the trace source, and one from a reference source, and determine whether
they are from the same source or not.
We already acquired measurements on single glass fragments.
So we `pair`_ the instances to get some pairs to compare.

.. _pair: reference.html#pairing-methods

.. jupyter-execute::

    from lir.transform.pairing import SourcePairing

    # combine the instances into pairs
    pairing_method = SourcePairing(ratio_limit=1)
    pairs = pairing_method.pair(glass_data, n_ref_instances=1, n_trace_instances=1)

    print(f'We have combined the {len(glass_data)} instances into {len(pairs)} pairs.')
    print(f'The paired data has features of all instances in the pairs, so it has type {type(pairs)}.')
    print(f'Of {len(pairs[pairs.labels==1])} pairs, both instances are from the same source.')
    print(f'Of {len(pairs[pairs.labels==0])} pairs, both instances are from different sources.')


Some LR systems may be able to work with multiple and trace reference in each pair.

.. jupyter-execute::

    from lir.transform.pairing import SourcePairing

    # combine the instances into pairs
    pairing_method = SourcePairing(ratio_limit=1)
    pairs_3x2 = pairing_method.pair(glass_data, n_trace_instances=2, n_ref_instances=3)

    print(f'We have combined the {len(glass_data)} instances into {len(pairs_3x2)} pairs.')
    print(f'The paired data has features of all instances in the pairs, so it has type {type(pairs_3x2)}.')
    print(f'Of {len(pairs_3x2[pairs_3x2.labels==1])} pairs, both instances are from the same source.')
    print(f'Of {len(pairs_3x2[pairs_3x2.labels==0])} pairs, both instances are from different sources.')


The actual comparison can take many forms, including a distance or similarity function such as the
`Manhattan distance`_.

.. _Manhattan distance: api/lir.transform.html#lir.transform.distance.ManhattanDistance

.. jupyter-execute::

    from lir.transform.distance import ManhattanDistance
    from lir.algorithms.logistic_regression import LogitCalibrator

    # reduce the pairs to a single value by calculating the Manhattan distance
    distances = ManhattanDistance().apply(pairs)

    print(f'The set of distances has type {type(distances)}.')
    print(f'The distances have an average of {np.mean(distances.features)}.')
    print(f'The standard deviation is {np.std(distances.features)}.')

Now it is time to calculate LLRs...

.. jupyter-execute::

    # calculate LLRs
    llrs = LogitCalibrator().fit_apply(distances)
    different_source_llrs = llrs[llrs.labels==0]
    same_source_llrs = llrs[llrs.labels==1]

    print(f'The set of LLRs has type {type(llrs)}.')
    print(f'The median LLR for different-source pairs is {np.median(different_source_llrs.llrs)}.')
    print(f'The median LLR for same-source pairs is {np.median(same_source_llrs.llrs)}.')


Data strategies
---------------

Above, we training the system and calculated LLRs using the same pairs, which is **not** a sound experimental setup!
In an experiment we work with `data strategies`_. This can be a simple train/test split, or a more
advanced configuration such as cross-validation.

A data strategy inherits from `DataStrategy`_ and implements an ``apply()`` method that returns an iterator of pairs of training and test sets.

.. _data strategies: api/lir.data.html#module-lir.data.data_strategies
.. _DataProvider: api/lir.data.html#lir.data.models.DataProvider
.. _DataStrategy: api/lir.data.html#lir.data.models.DataStrategy
.. _Pipeline: api/lir.transform.html#lir.transform.pipeline.Pipeline

Example:

.. jupyter-execute::

    from lir.data.data_strategies import MulticlassTrainTestSplit

    splitter = MulticlassTrainTestSplit(test_size=0.5)
    for training_data, test_data in splitter.apply(glass_data):
        print(f'We have {len(training_data)} instances available for training our models.')
        print(f'We have {len(test_data)} instances available as test data.')


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


A simple LR system could take the manhattan distance of each pair, apply some kind of calibration. We can combine those
steps in a pipeline.

.. jupyter-execute::

    from lir.transform.distance import ManhattanDistance
    from lir.algorithms.logistic_regression import LogitCalibrator

    comparing_pipeline = Pipeline([
        ('distance', ManhattanDistance()),
        ('logit', LogitCalibrator()),
    ])

    llrs = comparing_pipeline.fit_apply(pairs)


We can use these components in a score-based LR system:

.. jupyter-execute::

    from lir.lrsystems.score_based import ScoreBasedSystem

    lrsystem = ScoreBasedSystem(preprocessing_pipeline=preprocessing_pipeline, pairing_function=pairing_method, evaluation_pipeline=comparing_pipeline)
    for training_data, test_data in splitter.apply(glass_data):
        llrs = lrsystem.fit(training_data).apply(test_data)

    # plot results
    import lir.plotting
    with lir.plotting.show() as ax:
        ax.lr_histogram(llrs)

    # zoom in on the LLRs around 0
    with lir.plotting.show() as ax:
        ax.xlim(-8, 3)
        ax.lr_histogram(llrs, bins=100)

Judging from the results, this LR system does a poor job recognizing same-source pairs. Can you do better?


.. _LR system architectures: reference.html#lrsystem-architecture
.. _selection guide: reference.html#lrsystem_yaml
