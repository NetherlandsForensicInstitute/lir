import logging
from pathlib import Path

from lir.aggregation.base import Aggregation, AggregationData
from lir.config.base import ContextAwareDict, check_is_empty, config_parser, pop_field
from lir.config.data import parse_data_provider
from lir.data.io import DataFileBuilderCsv
from lir.data.models import DataProvider, FeatureData, check_type


LOG = logging.getLogger(__name__)


class CaseLLRToCsv(Aggregation):
    """
    Aggregation that applies a full-data-fitted LR system to case data and stores LLRs as CSV.

    Parameters
    ----------
    case_data_provider : DataProvider
        Provider for the case data to apply the LR system to.
    filename : str, optional
        Name of the output CSV file, by default 'case_llr.csv'.
    """

    def __init__(self, case_data_provider: DataProvider, filename: str = 'case_llr.csv') -> None:
        self.case_data_provider = case_data_provider
        self.filename = Path(filename)

    def report(self, data: AggregationData) -> None:
        """
        Apply the full-data-fitted LR system to the case data and store the resulting LLRs as CSV.

        Parameters
        ----------
        data : AggregationData
            Aggregation data containing the fitted LR system and case data.
        """
        if data.get_full_fit_lrsystem is not None:
            lrsystem = data.get_full_fit_lrsystem()
        else:
            LOG.warning(
                f'No full-data-fitted model factory available for run `{data.run_name}`; '
                f'using split-trained model instead.'
            )
            lrsystem = data.lrsystem

        # Ensure the case data does not contain labels by setting them to None.
        case_instances = self.case_data_provider.get_instances().replace(hypothesis=None)
        case_instances = check_type(FeatureData, case_instances)
        case_llrs = lrsystem.apply(case_instances)

        path = data.resolve_path_for_run(self.filename)

        if len(case_instances) != len(case_llrs):
            raise ValueError(
                f'Cannot export original case features to case_llr.csv because row counts differ: '
                f'{len(case_instances)} case rows vs {len(case_llrs)} LLR rows.'
            )

        csv_builder = DataFileBuilderCsv(path)

        features_2d = case_instances.features.reshape(case_instances.features.shape[0], -1)
        feature_count = features_2d.shape[1]
        raw_header = getattr(case_instances, 'header', None)
        if raw_header is not None and len(raw_header) == feature_count:
            feature_headers: list[str] = [str(h) for h in raw_header]
        else:
            feature_headers = [f'feature_{i}' for i in range(feature_count)]
        csv_builder.add_column(features_2d, dimension_headers={1: feature_headers})

        csv_builder.add_column(case_llrs.llrs, 'llr')

        if case_llrs.has_intervals and case_llrs.llr_intervals is not None:
            csv_builder.add_column(case_llrs.llr_intervals[:, 0], 'llr_interval_low')
            csv_builder.add_column(case_llrs.llr_intervals[:, 1], 'llr_interval_high')

        csv_builder.write()


@config_parser
def parse(config: ContextAwareDict, output_dir: Path) -> CaseLLRToCsv:
    """
    Parse output configuration for case LLR generation and CSV export.

    Parameters
    ----------
    config : ContextAwareDict
        Configuration dictionary containing case LLR output settings.
    output_dir : Path
        Directory where the CSV file will be written.

    Returns
    -------
    CaseLLRToCsv
        Configured CaseLLRToCsv aggregation instance.
    """
    case_data_provider = parse_data_provider(pop_field(config, 'case_llr_data'), output_dir)
    filename = pop_field(config, 'filename', default='case_llr.csv', validate=str)
    filename = check_type(str, filename)
    check_is_empty(config)
    return CaseLLRToCsv(case_data_provider, filename)
