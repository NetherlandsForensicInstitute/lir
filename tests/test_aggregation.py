import csv

import numpy as np
import pytest
from _pytest.tmpdir import TempPathFactory

from lir import registry
from lir.aggregation import Aggregation, AggregationData, CopyCSV, SubsetAggregation
from lir.config.base import GenericConfigParser, _expand
from lir.data.models import LLRData
from lir.lrsystems.binary_lrsystem import BinaryLRSystem
from lir.transform import Identity


def test_registry_items_available(synthesized_llrs_with_interval: LLRData, tmp_path_factory: TempPathFactory):
    """Test all registered output aggregation methods."""

    # create a temporary CSV file to use as the source for copy_csv
    source_csv = tmp_path_factory.mktemp('source') / 'source.csv'
    with open(source_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['a', 'b'])
        writer.writerow([1, 2])

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
        'output.copy_csv': {'source_file': str(source_csv)},
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
            assert isinstance(obj, Aggregation), (
                f'registry item is not an instance of `Aggregation`: {name}; found: {type(obj)}'
            )

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


def test_copy_csv_raises_on_missing_file(tmp_path):
    """CopyCSV should raise FileNotFoundError with a clear message when the source file does not exist."""
    missing = tmp_path / 'nonexistent.csv'
    with pytest.raises(FileNotFoundError, match="CopyCSV: File to copy"):
        CopyCSV(missing, tmp_path)


def test_copy_csv_copies_file(tmp_path):
    """CopyCSV should copy the source CSV file to the output directory on close."""
    source = tmp_path / 'source.csv'
    with open(source, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        writer.writerow([1, 2])

    output_dir = tmp_path / 'output'
    agg = CopyCSV(source, output_dir)
    agg.close()

    dest = output_dir / 'source.csv'
    assert dest.exists()
    with open(dest, newline='') as f:
        rows = list(csv.reader(f))
    assert rows == [['x', 'y'], ['1', '2']]


def test_copy_csv_filters_columns(tmp_path):
    """CopyCSV should only include specified columns when copying."""
    source = tmp_path / 'source.csv'
    with open(source, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'z'])
        writer.writerow([1, 2, 3])

    output_dir = tmp_path / 'output'
    agg = CopyCSV(source, output_dir, columns=['x', 'z'])
    agg.close()

    dest = output_dir / 'source.csv'
    with open(dest, newline='') as f:
        rows = list(csv.reader(f))
    assert rows == [['x', 'z'], ['1', '3']]


def test_copy_csv_new_file_name(tmp_path):
    """CopyCSV should use the provided new_file_name for the output file."""
    source = tmp_path / 'source.csv'
    source.write_text('a,b\n1,2\n')

    output_dir = tmp_path / 'output'
    agg = CopyCSV(source, output_dir, new_file_name='renamed.csv')
    agg.close()

    assert (output_dir / 'renamed.csv').exists()
    assert not (output_dir / 'source.csv').exists()
