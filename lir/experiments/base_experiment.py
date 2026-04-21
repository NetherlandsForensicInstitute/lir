from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from lir.aggregation import Aggregation


class Experiment(ABC):
    """
    Representation of an experiment pipeline run for each provided LR system.

    Parameters
    ----------
    name : str
        Name used to identify this object in outputs and logs.
    outputs : Sequence[Aggregation]
        Output aggregation definitions executed after each run.
    output_path : Path
        Path where generated outputs are written.
    """

    def __init__(
        self,
        name: str,
        outputs: Sequence[Aggregation],
        output_path: Path,
    ):
        self.name = name
        self.outputs = outputs
        self.output_path = output_path

    @abstractmethod
    def _generate_and_run(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        """
        Run experiment the configured experiment(s).

        This method ensures that all outputs are properly closed after the experiment run.
        """
        try:
            self._generate_and_run()
        finally:
            for output in self.outputs:
                output.close()
