from typing import Any

import numpy as np
from numpy import floating

from lir import LLRData
from lir.util import check_type


def count(llr_data: LLRData) -> float:
    """
    Count the number of instances or pairs of instances.

    Parameters
    ----------
    llr_data : LLRData
        LLRs and their metadata, wrapped in an ``LLRData`` object.

    Returns
    -------
    float
        The number of instances or pairs of instances.
    """
    return len(llr_data)


def count_h1(llr_data: LLRData) -> float:
    """
    Count the number of instances or pairs of instances labeled as Hypothesis 1 (hypothesis label 1).

    Parameters
    ----------
    llr_data : LLRData
        LLRs and their metadata, wrapped in an ``LLRData`` object.

    Returns
    -------
    float
        The number of instances or pairs of instances.
    """
    return np.sum(check_type(np.ndarray, llr_data.labels) == 1)


def count_h2(llr_data: LLRData) -> float:
    """
    Count the number of instances or pairs of instances labeled as Hypothesis 2 (hypothesis label 0).

    Parameters
    ----------
    llr_data : LLRData
        LLRs and their metadata, wrapped in an ``LLRData`` object.

    Returns
    -------
    float
        The number of instances or pairs of instances.
    """
    return np.sum(llr_data.require_labels == 0)


def mean_llr_h1(llr_data: LLRData) -> floating[Any]:
    """
    Calculate the average LLR of instances (or instance pairs) labeled as Hypothesis 1 (hypothesis label 1).

    Parameters
    ----------
    llr_data : LLRData
        LLRs and their metadata, wrapped in an ``LLRData`` object.

    Returns
    -------
    float
        The average LLR.
    """
    return np.mean(llr_data.llrs[llr_data.require_labels == 1])


def mean_llr_h2(llr_data: LLRData) -> floating[Any]:
    """
    Calculate the average LLR of instances (or instance pairs) labeled as Hypothesis 2 (hypothesis label 0).

    Parameters
    ----------
    llr_data : LLRData
        LLRs and their metadata, wrapped in an ``LLRData`` object.

    Returns
    -------
    float
        The average LLR.
    """
    return np.mean(llr_data.llrs[llr_data.require_labels == 0])
