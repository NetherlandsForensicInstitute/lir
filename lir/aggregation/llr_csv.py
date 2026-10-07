import logging
from pathlib import Path

from lir.aggregation.base import Aggregation, AggregationData
from lir.config.base import ConfigValue, check_is_empty, config_parser, pop_field
from lir.data.io import DataFileBuilderCsv
from lir.util import check_not_none


LOG = logging.getLogger(__name__)


class LLRToCsv(Aggregation):
    """
    Aggregation that writes the LLR data of a run to a CSV file.

    The CSV contains the LLRs, and the confidence intervals if available, optionally preceded by the
    source IDs.

    Parameters
    ----------
    filename : str, optional
        Name of the output CSV file, by default 'case_llr.csv'.
    """

    def __init__(self, filename: str = 'case_llr.csv') -> None:
        self.filename = Path(filename)

    def report(self, data: AggregationData) -> None:
        """
        Write the LLR data of the run to a CSV file.

        Parameters
        ----------
        data : AggregationData
            Aggregation data containing the LLR data to write.
        """
        llrdata = check_not_none(data.llrdata)

        path = data.resolve_path_for_run(self.filename)

        csv_builder = DataFileBuilderCsv(path)

        if llrdata.source_ids is not None:
            csv_builder.add_column(llrdata.source_ids, 'source_id')

        csv_builder.add_column(llrdata.llrs, 'llr')

        if llrdata.has_intervals and llrdata.llr_intervals is not None:
            csv_builder.add_column(llrdata.llr_intervals[:, 0], 'llr_interval_low')
            csv_builder.add_column(llrdata.llr_intervals[:, 1], 'llr_interval_high')

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
        Configured LLRToCsv aggregation instance.
    """
    filename = pop_field(config, 'filename', default='case_llr.csv', validate_type=str)
    check_is_empty(config)
    return LLRToCsv(filename)
