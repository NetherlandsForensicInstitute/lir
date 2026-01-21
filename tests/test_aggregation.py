import pytest
from _pytest.tmpdir import TempPathFactory

from lir import registry
from lir.aggregation import Aggregation, AggregationData
from lir.config.base import GenericConfigParser, _expand
from lir.data.models import LLRData
from lir.lrsystems.binary_lrsystem import BinaryLRSystem
from lir.transform import Identity


def test_registry_items_available(synthesized_llrs_with_interval: LLRData, tmp_path_factory: TempPathFactory):
    """Test all registered output aggregation methods."""

    # define a mapping from output aggregator to initialization arguments
    args = {'output.csv': {'columns': []}}

    # iterate over all registry items
    for name in registry.registry():
        # test aggregators within the output section only
        if name.startswith('output.'):
            # create the object
            parser = registry.get(name, default_config_parser=GenericConfigParser)
            args = _expand([], args.get(name, {}))
            obj = parser.parse(args, tmp_path_factory.mktemp('output'))
            assert isinstance(obj, Aggregation), f'registry item is not an instance of `Aggregation`: {name}'

            # generate output
            try:
                lrsystem = BinaryLRSystem(pipeline=Identity())
                obj.report(
                    AggregationData(llrdata=synthesized_llrs_with_interval, lrsystem=lrsystem, parameters_str='')
                )
            except Exception as _:
                pytest.fail(f'generating output failed for registry item `{name}`')
