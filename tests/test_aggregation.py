import numpy as np
import pytest
from _pytest.tmpdir import TempPathFactory

from lir import registry
from lir.aggregation import Aggregation, AggregationData, SubsetAggregation
from lir.config.base import GenericConfigParser, _expand
from lir.data.models import LLRData
from lir.lrsystems.binary_lrsystem import BinaryLRSystem
from lir.transform import Identity


def test_registry_items_available(synthesized_llrs_with_interval: LLRData, tmp_path_factory: TempPathFactory):
    """Test all registered output aggregation methods."""

    # define a mapping from output aggregator to initialization arguments
    args_by_method = {
        'output.csv': {'columns': []},
        'output.case_llr': {
            'case_llr_data': {
                'method': 'synthesized_normal_binary',
                'seed': 42,
                'h1': {'mean': 1, 'std': 1, 'size': 10},
                'h2': {'mean': -1, 'std': 1, 'size': 10},
            }
        },
        'output.by_category': {'category_field': 'my_category_field', 'output': 'pav'},
    }

    synthesized_llrs_with_interval = synthesized_llrs_with_interval.replace(
        my_category_field=np.array(['a'] * len(synthesized_llrs_with_interval))
    )

    # iterate over all registry items
    for name in registry.registry():
        # test aggregators within the output section only
        if name.startswith('output.'):
            # create the object
            parser = registry.get(name, default_config_parser=GenericConfigParser)
            args = _expand([], args_by_method.get(name, {}))
            obj = parser.parse(args, tmp_path_factory.mktemp('output'))
            assert isinstance(obj, Aggregation), f'registry item is not an instance of `Aggregation`: {name}'

            # generate output
            try:
                lrsystem = BinaryLRSystem(pipeline=Identity())
                obj.report(
                    AggregationData(
                        llrdata=synthesized_llrs_with_interval, lrsystem=lrsystem, parameters={}, run_name=''
                    )
                )
            except Exception as _:
                pytest.fail(f'generating output failed for registry item `{name}`')


def test_subset_aggregation():
    class MyAggregation(Aggregation):
        def report(self, data: AggregationData) -> None:
            assert len(data.llrdata) == len(llrs) / 2, 'number of LLRs within a category'
            assert np.all(data.llrdata.llrs == data.llrdata.llrs[0]), (
                'LLRs of the same category must have the same value'
            )

    llrs = LLRData(features=np.arange(2).repeat(10).reshape((20, 1)), category=np.arange(2).repeat(10))
    aggregation = SubsetAggregation(aggregation_methods=[MyAggregation()], category_field='category')
    aggregation.report(
        AggregationData(run_name='testrun', llrdata=llrs, lrsystem=BinaryLRSystem(pipeline=Identity()), parameters={})
    )
