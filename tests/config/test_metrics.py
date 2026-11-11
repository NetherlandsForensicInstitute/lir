import csv
from pathlib import Path

import confidence
import numpy as np
import pytest

from lir import metrics
from lir.aggregation.base import AggregationData
from lir.aggregation.metrics_csv import parse as parse_metrics_csv
from lir.config.base import ConfigValue, YamlParseError
from lir.config.metrics import parse_individual_metric, parse_metrics
from lir.data.models import LLRData


OUTPUT_PATH = Path('/output')
CONTEXT = ['experiments', '0', 'output', '0', 'columns']
LLR_DATA = LLRData(features=np.array([2.0, -1.0]), hypothesis=np.array([1, 0]))


def test_parse_metric_by_name():
    assert parse_individual_metric(ConfigValue.wrap(CONTEXT, 'cllr'), OUTPUT_PATH) is metrics.cllr


def test_parse_metric_with_arguments():
    metric = parse_individual_metric(ConfigValue.wrap(CONTEXT, {'method': 'cllr', 'h1_weight': 0.25}), OUTPUT_PATH)
    assert metric(LLR_DATA) == pytest.approx(metrics.cllr(LLR_DATA, 0.25))


def test_parse_metric_with_unknown_argument():
    with pytest.raises(YamlParseError, match='unrecognized argument: bogus'):
        parse_individual_metric(ConfigValue.wrap(CONTEXT, {'method': 'cllr', 'bogus': 1}), OUTPUT_PATH)


def test_parse_metric_with_unknown_name():
    with pytest.raises(YamlParseError, match='component not found'):
        parse_individual_metric(ConfigValue.wrap(CONTEXT, 'nonexistent_metric'), OUTPUT_PATH)


def test_parse_metric_without_method():
    with pytest.raises(YamlParseError, match='missing field: `method`'):
        parse_individual_metric(ConfigValue.wrap(CONTEXT, {'h1_weight': 0.5}), OUTPUT_PATH)


def test_parse_metric_with_invalid_type():
    with pytest.raises(YamlParseError, match='expected a mapping with a `method` field'):
        parse_individual_metric(ConfigValue.wrap(CONTEXT, 42), OUTPUT_PATH)


def test_parse_metrics():
    parsed = parse_metrics(
        ConfigValue.wrap(
            CONTEXT,
            [
                'cllr',
                {'name': 'cllr_forensic', 'method': 'cllr', 'h1_weight': 0.9},
                {'method': 'cllr_min'},
            ],
        ),
        OUTPUT_PATH,
    )

    assert list(parsed) == ['cllr', 'cllr_forensic', 'cllr_min']
    assert parsed['cllr'] is metrics.cllr
    assert parsed['cllr_forensic'](LLR_DATA) == pytest.approx(metrics.cllr(LLR_DATA, 0.9))
    assert parsed['cllr_min'] is metrics.cllr_min


def test_parse_metrics_with_duplicate_names():
    with pytest.raises(YamlParseError, match='duplicate metric name: `cllr`'):
        parse_metrics(
            ConfigValue.wrap(CONTEXT, ['cllr', {'method': 'cllr', 'h1_weight': 0.9}]),
            OUTPUT_PATH,
        )


def test_metrics_csv_with_weighted_metric(tmp_path: Path):
    config = ConfigValue.wrap(
        [],
        confidence.loads(
            """
            columns:
              - cllr
              - name: cllr_forensic
                method: cllr
                h1_weight: 0.9
            """
        ),
    )

    aggregation = parse_metrics_csv(config, tmp_path)
    try:
        aggregation.report(
            AggregationData(
                llrdata=LLR_DATA,
                lrsystem=None,
                parameters={},
                run_name='1',
                experiment_output_dir=tmp_path,
                run_output_dir=tmp_path,
            )
        )
    finally:
        aggregation.close()

    with open(tmp_path / 'metrics.csv') as f:
        row = next(csv.DictReader(f))

    assert set(row) == {'cllr', 'cllr_forensic'}
    assert float(row['cllr']) == pytest.approx(metrics.cllr(LLR_DATA))
    assert float(row['cllr_forensic']) == pytest.approx(metrics.cllr(LLR_DATA, 0.9))
