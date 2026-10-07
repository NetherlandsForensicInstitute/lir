import numpy as np
import pytest
from _pytest.tmpdir import TempPathFactory

from lir import registry
from lir.aggregation import Aggregation, AggregationData, LLRToCsv
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


def test_llr_to_csv_report(synthesized_llrs_with_interval: LLRData, tmp_path_factory: TempPathFactory):
    """Test that `LLRToCsv.report` writes the LLRs and intervals to a CSV file."""
    n = len(synthesized_llrs_with_interval)
    llrdata = synthesized_llrs_with_interval.replace(source_ids=np.array([f'id_{i}' for i in range(n)]))

    output_dir = tmp_path_factory.mktemp('output')
    aggregator = LLRToCsv()
    aggregator.report(
        AggregationData(
            llrdata=llrdata,
            lrsystem=None,
            parameters={},
            run_name='',
            experiment_output_dir=output_dir,
            run_output_dir=output_dir,
        )
    )

    csv_path = output_dir / 'case_llr.csv'
    assert csv_path.is_file()
    lines = csv_path.read_text().strip().splitlines()
    assert lines[0] == 'source_id,llr,llr_interval_low,llr_interval_high'
    assert len(lines) == n + 1
    assert lines[1].startswith('id_0,')


def test_llr_to_csv_report_requires_llrdata(tmp_path_factory: TempPathFactory):
    """Test that `LLRToCsv.report` raises when the LLR data is missing."""
    output_dir = tmp_path_factory.mktemp('output')
    aggregator = LLRToCsv()
    with pytest.raises(ValueError):
        aggregator.report(
            AggregationData(
                llrdata=None,
                lrsystem=None,
                parameters={},
                run_name='',
                experiment_output_dir=output_dir,
                run_output_dir=output_dir,
            )
        )
