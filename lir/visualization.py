from pathlib import Path

import numpy as np

from lir.plotting import savefig


def pav(base_path: Path, llrs: np.ndarray, labels: np.ndarray, **kwargs) -> None:
    """Helper function to generate and save a PAV-plot to the output directory."""
    base_path.mkdir(exist_ok=True, parents=True)
    path = base_path / "pav.png"
    with savefig(str(path)) as fig:
        fig.pav(llrs, labels, **kwargs)


def ece(base_path: Path, llrs: np.ndarray, labels: np.ndarray, **kwargs) -> None:
    """Helper function to generate and save an ECE-plot to the output directory."""
    base_path.mkdir(exist_ok=True, parents=True)
    path = base_path / "ece.png"
    with savefig(str(path)) as fig:
        fig.ece(llrs, labels, **kwargs)


def lr_histogram(base_path: Path, llrs: np.ndarray, labels: np.ndarray, **kwargs) -> None:
    """Helper function to generate and save an histogram-plot to the output directory."""
    base_path.mkdir(exist_ok=True, parents=True)
    path = base_path / "histogram.png"
    with savefig(str(path)) as fig:
        fig.lr_histogram(llrs, labels, **kwargs)
