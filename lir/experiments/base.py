from abc import ABC, abstractmethod
from pathlib import Path


class Experiment(ABC):
    """
    Representation of an experiment to evaluate LR systems.

    Parameters
    ----------
    output_path : Path
        Path where generated outputs are written.
    """

    def __init__(
        self,
        output_path: Path,
    ):
        self.output_path = output_path

    @abstractmethod
    def run(self) -> None:
        """Execute the experiment."""
        raise NotImplementedError
