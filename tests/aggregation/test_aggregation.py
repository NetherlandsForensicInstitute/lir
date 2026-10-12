import numpy as np
import pytest
from _pytest.tmpdir import TempPathFactory

from lir import registry
from lir.aggregation import Aggregation, AggregationData
from lir.config import check_is_empty
from lir.config.base import ConfigValue, GenericConfigParser
from lir.data.models import LLRData
from lir.lrsystems.binary_lrsystem import BinaryLRSystem
from lir.transform import Identity


def test_registry_items_available(synthesized_llrs_with_interval: LLRData, tmp_path_factory: TempPathFactory):
    """Test all registered output aggregation methods."""

    # define a mapping from output aggregator to initialization arguments
    args_by_method = ConfigValue.wrap(
        [],
        {
            'output.metrics_csv': {'columns': []},
            'output.llr_csv': {},
            'output.by_category': {'category_field': 'my_category_field', 'output': 'pav'},
        },
    )

    synthesized_llrs_with_interval = synthesized_llrs_with_interval.replace(
        my_category_field=np.array(['a'] * len(synthesized_llrs_with_interval))
    )

    # iterate over all registry items
    for name in registry.registry():
        # test aggregators within the output section only
        if name.startswith('output.'):
            # create the object
            parser = registry.get(name, default_config_parser=GenericConfigParser)
            output_dir = tmp_path_factory.mktemp('output')
            obj = parser.parse(args_by_method.pop(name, default={}), output_dir)  # type: ignore
            assert isinstance(obj, Aggregation), (
                f'registry item is not an instance of `Aggregation`: {name}; found: {type(obj)}'
            )

            # generate output
            try:
                lrsystem = BinaryLRSystem(pipeline=Identity())
                obj.report(
                    AggregationData(
                        llrdata=synthesized_llrs_with_interval,
                        lrsystem=lrsystem,
                        parameters={},
                        run_name='',
                        experiment_output_dir=output_dir,
                        run_output_dir=output_dir,
                    )
                )
            except Exception as _:
                pytest.fail(f'generating output failed for registry item `{name}`')
            finally:
                obj.close()

    check_is_empty(args_by_method)
