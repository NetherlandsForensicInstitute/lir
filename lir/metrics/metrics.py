from lir import LLRData
from lir.aggregation import AggregationData
from lir.metrics.base import aggregation_metric, llr_metric


@llr_metric
def llr_upper_bound(llrs: LLRData) -> float | None:
    """
    Provide corresponding upper bound for provided LLR data.

    When an LLRData object contains an upper bound, return it. If not, return None.

    Parameters
    ----------
    llrs : LLRData
        LLRs and their metadata, wrapped in an `LLRData` object.

    Returns
    -------
    float | None
        The LLR upper bound, or `None`.
    """
    return llrs.llr_upper_bound


@llr_metric
def llr_lower_bound(llrs: LLRData) -> float | None:
    """
    Provide corresponding lower bound for provided LLR data.

    When an LLRData object contains a lower bound, return it. If not, return None.

    Parameters
    ----------
    llrs : LLRData
        LLRs and their metadata, wrapped in an `LLRData` object.

    Returns
    -------
    float | None
        The LLR lower bound, or `None`.
    """
    return llrs.llr_lower_bound


@aggregation_metric
def runtime(data: AggregationData) -> float:
    """
    Get the total runtime of a run from an ``AggregationData`` object.

    Parameters
    ----------
    data : AggregationData
        An AggregationData object.

    Returns
    -------
    float
        The runtime in seconds.
    """
    return data.runtime_secs
