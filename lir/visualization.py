from pathlib import Path

import numpy as np

from lir.data.models import LLRData
from lir.plotting import savefig


def pav(base_path: Path, llrs: LLRData, labels: np.ndarray) -> None:
    """Helper function to generate and save a PAV-plot to the output directory."""
    base_path.mkdir(exist_ok=True, parents=True)
    path = base_path / 'pav.png'
    with savefig(str(path)) as fig:
        fig.pav(llrs.llrs, labels)


def ece(base_path: Path, llrs: LLRData, labels: np.ndarray) -> None:
    """Helper function to generate and save an ECE-plot to the output directory."""
    base_path.mkdir(exist_ok=True, parents=True)
    path = base_path / 'ece.png'
    with savefig(str(path)) as fig:
        fig.ece(llrs.llrs, labels)


def lr_histogram(base_path: Path, llrs: LLRData, labels: np.ndarray) -> None:
    """Helper function to generate and save an histogram-plot to the output directory."""
    base_path.mkdir(exist_ok=True, parents=True)
    path = base_path / 'histogram.png'
    with savefig(str(path)) as fig:
        fig.lr_histogram(llrs.llrs, labels)


def llr_interval(base_path: Path, llrs: LLRData, labels: np.ndarray | None = None) -> None:
    """Helper function to generate and save a Score-LR plot to the output directory."""
    base_path.mkdir(exist_ok=True, parents=True)
    path = base_path / 'llr_interval.png'
    with savefig(str(path)) as fig:
        fig.llr_interval(llrs)
