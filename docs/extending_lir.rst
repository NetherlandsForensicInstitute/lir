Extending LiR by creating a custom component
============================================

LiR is designed to be easily extensible, allowing developers to add new features and functionality without modifying the core code.
This document outlines the process for creating a custom component that can be used in an experiment setup configuration file.

In this example, we will create a custom module that calculates the cosine similarity between two vectors.
The module will be integrated into the LiR framework and can be used in experiments just like any other built-in component.

This extends to other types of components, such as experiment types, LR system architectures, metrics, etc.
 

Step 1: setup a new project
---------------------------

Let's start by setting up a new project for an experiment with a simple LR system.
The following example shows a minimal configuration file for a single run experiment:

.. literalinclude:: snippets/minimal-single-run.yaml
    :language: yaml

To run this experiment, save this snippet as ``config.yaml`` and run it with the following command:

.. code-block:: bash

    lir run config.yaml


Step 2: implement cosine similarity
-----------------------------------

Create a file named ``cosine_similarity.py`` and implement the cosine similarity calculation as a class that inherits from :class:`~lir.Transformer`.

.. code-block:: python

    import numpy as np
    from numpy.linalg import norm
    from lir import Transformer, InstanceData, FeatureData, PairedFeatureData, check_type

    class CosineSimilarity(Transformer):
        def apply(self, pairs: InstanceData) -> FeatureData:
            # make sure that we have pairs of feature vectors
            pairs = check_type(PairedFeatureData, pairs)

            # flatten the feature vectors for the trace and reference data
            trace_features = pairs.trace_features.reshape(-1)
            ref_features = pairs.ref_features.reshape(-1)

            # calculate the similarities between the trace and the reference feature vectors
            cosine_similarities = np.dot(trace_features, ref_features) / (norm(trace_features) * norm(ref_features))

            # return a FeatureData object with the same attributes as the input pairs, but with the calculated similarities as the features
            return pairs.replace_as(FeatureData, features=cosine_similarities)

This module is now ready to run in an LR system, and can be referenced in a configuration file as ``cosine_similarity.CosineSimilarity``.

.. literalinclude:: snippets/minimal-single-run-with-cosim.yaml
    :language: yaml
    :lines: 1-32,34-
    :emphasize-lines: 31-32


Step 3: add parameters
----------------------

If the component has no parameters, you are done.
Add parameters by adding them to the ``__init__`` method. In this example, we will add a parameter to control whether the cosine similarity should be squared or not.

.. code-block:: python

    import numpy as np
    from numpy.linalg import norm
    from lir import Transformer, InstanceData, FeatureData, PairedFeatureData, check_type

    class CosineSimilarity(Transformer):
        def __init__(self, square: bool = False):
            self.square = square

        def apply(self, pairs: InstanceData) -> FeatureData:
            # make sure that we have pairs of feature vectors
            pairs = check_type(PairedFeatureData, pairs)

            # flatten the feature vectors for the trace and reference data
            trace_features = pairs.trace_features.reshape(-1)
            ref_features = pairs.ref_features.reshape(-1)

            # calculate the similarities between the trace and the reference feature vectors
            cosine_similarities = np.dot(trace_features, ref_features) / (norm(trace_features) * norm(ref_features))

            if self.square:
                cosine_similarities = np.square(cosine_similarities)

            # return a FeatureData object with the same attributes as the input pairs, but with the calculated similarities as the features
            return pairs.replace_as(FeatureData, features=cosine_similarities)

Any key/value-pair in the configuration section of ``scorer`` (except for ``method``) will be passed to the component when it is instantiated.

.. literalinclude:: snippets/minimal-single-run-with-cosim.yaml
    :language: yaml
    :emphasize-lines: 33


Step 4: create a configuration parser (optional)
------------------------------------------------

There is a good chance that this is all you need to do to create a new component. However, if you want more control over how the component is instantiated, you can create a custom configuration parser.

We create a configuration parser by defining a function that parses a configuration section and returns a ``CosineSimilarity`` object.
We take care that the function:
- takes two arguments, the configuration section in the YAML that is needed to initialize the module, and an output directory;
- returns an instance of the component (in our case, ``CosineSimilarity``);
- is marked as a configuration parser with the @:meth:`~lir.config.config_parser` decorator. 

.. code-block:: python

    import logging
    from lir.config import config_parser, ConfigValue

    LOG = logging.getLogger(__name__)

    @config_parser
    def parse_cosine_similarity_config(config: ConfigValue, output_dir: Path) -> CosineSimilarity:
        LOG.debug('parsing the configuration section at: ' + '.'.join(config.context))

        # obtain the value of the "square" parameter from the configuration dictionary, and remove it from the dictionary.
        square = pop_field(config, "square", default=False)

        # check that there are no other parameters left unparsed, and raise an error if there are.
        check_is_empty(config)

        # instantiate the component with the parsed parameters and return it.
        return CosineSimilarity(square=square)

The ``config`` parameter contains the relevant section in the configuration YAML.
Each :class:`~lir.config.ConfigValue` object has a ``context`` attribute, which is the full path of the configuration value in the original YAML file, and a ``value`` attribute, which is the actual value.
If ``value`` is a container object such as a list or a dictionary, it is also wrapped inside a :class:`~lir.config.ConfigValue` object.
Please refer to the API documentation for more information on dealing with :class:`~lir.config.ConfigValue` objects.

In our case, the ``config`` parameter holds the configuration section under ``score`` (line 31), which is a dictionary of two keys, namely ``method`` and ``square``.
Since the ``method`` is already consumed to identify the configuration parser, this leaves a dictionary with just ``square`` as its single key.

The ``output_dir`` value is the path to the directory where the component may save its results, if applicable.

To use this configuration parser, simply replace ``CosineSimilarity`` in the configuration file in `Step 3: add parameters`_ with ``parse_cosine_similarity_config``.


Step 5: register your component (optional)
------------------------------------------

Frequently used components can be registered with LiR, so that they can be referenced in configuration files without needing to specify the full module path.
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

Other components can be added in a similar way, by creating a function with the ``@config_parser`` decorator, and returning the component instance.


A data provider to read from a database
---------------------------------------

We will create a data provider that obtains its data from a Sqlite database.

We'll first create and populate a database with dummy values.

.. code-block:: python

    import sqlite3

    with sqlite3.open('example.db') as db:
        sqlite3.execute('CREATE TABLE feature_table (hypothesis INT, feature1 FLOAT, feature2 FLOAT)')
        sqlite3.execute('INSERT INTO feature_table VALUES(1, 1.0, 1.1), (1, 1.1, 1.2), (0, 3.1, 3.3), (0, 3.0, 4.0)')


Now that we have the database, we can create a data provider component to read it.
Here, we demonstrate three ways to accomplish that.
The recommended and most simple way to do it, is to define a function that returns the data, and decorate it with @:meth:`~lir.config.data_provider`:

.. code-block:: python

    import sqlite3
    from lir import FeatureData, DataProvider
    from lir.config import ConfigValue

    @data_provider
    def read_from_sqlite3(path: str) -> FeatureData:
        with sqlite3.open(path) as db:
            hypotheses = []
            features = []

            result = db.execute('SELECT feature1, feature2 FROM feature_table')
            for row in result:
                hypotheses.append(row[0])
                features.append([row[1], row[2]])

            return FeatureData(hyptheses=np.array(hypothesis), features=np.array(features))


Alternatively, we may define a :class:`~lir.DataProvider` class. If its ``__init__`` method requires no parameters or atomic parameters only, like here, it can be used directly.
Otherwise, we'll have to use @:meth:`lir.config.config_parser`:

.. code-block:: python

    class SqliteDataProvider(DataProvider):
        def __init__(path: str):
            self.path = path
        
        def get_instances() -> FeatureData:
            return read_from_sqlite3(self.path)
    
    @config_parser
    def parse_sqlite_data_provider_config(config: ConfigValue, output_dir: Path) -> SqliteDataProvider:
        path = pop_field(config, 'path')
        return SqliteDataProvider(path)


Data count experiment
---------------------

This example shows how to create a new type of experiment. All it does is to count the number of instances in a dataset.

.. code-block:: python

    class DataCounterExperiment(Experiment):
        def __init__(data_provider: DataProvider, output_file: Path):
            self.data_provider = data_provider
            self.output_file = output_file

        def run():
            number_of_instances = len(self.data_provider.get_instances())
            with open(output_file) as f:
                f.write(f'{number_of_instances}\n')


    @config_parser
    def parse_data_counter_experiment_config(config: ConfigValue, output_dir: Path) -> Experiment:
        data_provider_config = pop_value('data_provider')
        data_provider = parse_data_provider(data_provider_config)
        return DataCounterExperiment(data_provider, output_dir / 'count.txt')


Output the average LLR
----------------------

The ``output`` section of the experiments list methods to aggregate results.
This is done by :class:`~lir.Aggregation` objects, which receive the results of each LR system run one by one.

In this example, we will create an output method that calculates the average LLR for each run, and writes it to a file.
Finally, we will calculate the average of all runs.

.. code-block:: python

    class AverageLLR(Aggregation):
        def __init__(self, path: Path):
            self.path = path
            self._average_cumulative = 0
            self._average_count = 0

        def report(self, aggregation_data: AggregationData):
            average_llr = np.average(aggregation_data.llrdata.llrs)
            with open(self.path, 'w') as f:
                f.write(f'{average_llr}\n')

            self._average_cumulative += average_llr
            self._average_count += 1
        
        def close(self):
            with open(self.path, 'w') as f:
                f.write(f'{self._average_cumulative / self._average_count}\n')

    @config_parser
    def average_llr(config: ConfigValue, output_dir: Path) -> AverageLLR:
        filename = pop_field(config, 'filename')
        return AverageLLR(output_dir / filename)


A new metric
------------

A metric is a function that takes :class:`~lir.LLRData` and returns a ``float``.
In this example, we implement the average of LLRs as a metric.

.. code-block:: python

    import functools

    def average_llr(llrdata: LLRData) -> float:
        return np.average(llrdata.llrs)


A slightly more complicated example is the calculation of a weighted CLLR value, which involves the @:meth:`~lir.config.config_parser` decorator.

.. code-block:: python

    def calculate_weighted_cllr(h0_weight: float, h1_weight: float, llrdata: LLRData) -> float:
        lrs = logodds_to_odds(llrdata.llrs)
        lrs0 = lrs[llrdata.hypothesis == 0]
        lrs1 = lrs[llrdata.hypothesis == 1]
        cllr0 = h0_weight * np.mean(np.log2(1 + lrs0))
        cllr1 = h1_weight * np.mean(np.log2(1 + 1 / lrs1))
        return (cllr0 + cllr1) / (h0_weight + h1_weight)

    @config_parser
    def weighted_cllr(config: ConfigValue, output_dir: Path) -> Callable[[LLRData], float]:
        h0_weight = pop_field(config, 'h0_weight', validate=float, default=1.0)
        h1_weight = pop_field(config, 'h1_weight', validate=float, default=1.0)
        return functools.partial(calculate_weighted_cllr, h0_weight, h1_weight)


A data strategy that skips training
-----------------------------------

Some models do not require a training phase.
In this example, we implement a data strategy that assumes we work with such models only, and offers the full dataset for testing.

..code-block:: python

    class AllTest(DataStrategy):
        def apply[DataType: InstanceData](self, instances: DataType) -> Iterator[tuple[DataType, DataType]]:
            yield None, instances
