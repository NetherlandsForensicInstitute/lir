import logging
from pathlib import Path

from lir.aggregation.base import Aggregation, AggregationData
from lir.config.base import ConfigValue, check_is_empty, config_parser, pop_field
from lir.data.io import DataFileBuilderCsv


LOG = logging.getLogger(__name__)


class LLRToCsv(Aggregation):
    """
    Aggregation that applies a full-data-fitted LR system to case data and stores LLRs as CSV.

    Parameters
    ----------
    filename : str, optional
        Name of the output CSV file, by default 'case_llr.csv'.
    """

    def __init__(self, filename: str = 'case_llr.csv') -> None:
        self.filename = Path(filename)

    def report(self, data: AggregationData) -> None:
        """
        Apply the full-data-fitted LR system to the case data and store the resulting LLRs as CSV.

        Parameters
        ----------
        data : AggregationData
            Aggregation data containing the fitted LR system and case data.
        """
        path = data.resolve_path_for_run(self.filename)

        csv_builder = DataFileBuilderCsv(path)

        if data.llrdata.source_ids is not None:
            csv_builder.add_column(data.llrdata.source_ids, 'source_id')

        csv_builder.add_column(data.llrdata.llrs, 'llr')

        if data.llrdata.has_intervals and data.llrdata.llr_intervals is not None:
            csv_builder.add_column(data.llrdata.llr_intervals[:, 0], 'llr_interval_low')
            csv_builder.add_column(data.llrdata.llr_intervals[:, 1], 'llr_interval_high')

        csv_builder.write()


@config_parser
def parse(config: ConfigValue, output_dir: Path) -> LLRToCsv:
    """
    Parse output configuration for case LLR generation and CSV export.

    Parameters
    ----------
    config : ConfigValue
        Configuration dictionary containing case LLR output settings.
    output_dir : Path
        Directory where the CSV file will be written.

    Returns
    -------
    LLRToCsv
        Configured CaseLLRToCsv aggregation instance.
    """
    filename = pop_field(config, 'filename', default='case_llr.csv', validate_type=str)
    check_is_empty(config)
    return LLRToCsv(filename)
