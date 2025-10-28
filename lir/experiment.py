import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence, Callable
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

import lir
from lir.aggregation import Aggregation
from lir.data.models import DataStrategy, concatenate_instances
from lir.lrsystems.lrsystems import LRSystem, LLRData

LOG = logging.getLogger(__name__)


def np_concatenate_optional(optional_values: list[Any]) -> np.ndarray | None:
    """Helper function to use `np.concatenate()` using collections with optional None values.

    When a list may contain optional `None` values, it is not allowed (by type checking) to directly
    call `np.concatenate()` on this list, even when explicitly checked that this list does not contain
    `None` values.

    This helper function either directly returns `None` if a collection contains a `None` value, or
    it uses a workaround by using a list comprehension which filters out the `None` values (which are not
    present anymore) to satisfy the type checker.
    """
    if None in optional_values:
        return None

    # Workaround for type checking optional None values
    return np.concatenate([value for value in optional_values if value is not None])


class Experiment(ABC):
    """Representation of an experiment pipeline run for each provided LR system."""

    def __init__(
        self,
        name: str,
        data: DataStrategy,
        aggregations: Sequence[Aggregation],
        visualization_functions: list[Callable],
        output_path: Path,
    ):
        self.name = name
        self.data = data
        self.aggregations = aggregations
        self.visualization_functions = visualization_functions
        self.output_path = output_path

    def _run_lrsystem(self, lrsystem: LRSystem) -> LLRData:
        """Run experiment on a single LR system configuration using the provided data(setup).

        First, the data is split into a training and testing subset, according to the provided
        data setup strategy. Next, the LR system is fitted using the training subset data and
        subsequently used to determine LLRs for the test subset data. The results are stored in
        a temporary list which contains the determined data of each test / train split.

        Metrics are collected as specified in the `metrics` mapping and returned after the experiment run.
        Visual representations of the obtained results for the specific LR system are generated as specified
        through the `visualization_functions` and stored in the `output_path` directory.
        """
        # Placeholders for numpy array's of LLRs and labels obtained from each train/test split
        llr_sets: list[LLRData] = []

        # Split the data into a train / test subset, according to the provided DataSetup. This could
        # for example be a simple binary split or a multiple fold cross validation split.
        for training_data, test_data in self.data:
            lrsystem.fit(training_data)
            subset_llr_results: LLRData = lrsystem.apply(test_data)

            # Store results (numpy arrays) into the placeholder lists
            llr_sets.append(subset_llr_results)

        # Combine collected numpy array's after iteration over the train/test split(s)
        combined_llrs: LLRData = concatenate_instances(*llr_sets)
        llrs = combined_llrs.llrs
        labels = combined_llrs.labels

        # Generate visualization output as configured by `visualization_functions`
        # and write graphical output to the `output_path`.
        output_dir = self.output_path / lrsystem.name
        LOG.debug(f"writing visualizations to {output_dir}")
        for visualization_function in self.visualization_functions:
            visualization_function(output_dir, llrs, labels)

        # Construct a `results` dictionary of metrics indicating the performance of the given LR system
        for aggregation in self.aggregations:
            aggregation.report(llrs, labels, lrsystem.parameters)

        return combined_llrs

    @abstractmethod
    def _generate_and_run(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        """Run experiment for all configured LR systems.

        Perform the single experiment of all configured LR systems and write the obtained
        key metrics - results on the performance of the LR system - to the dedicated `metrics.csv`
        file in the `output_path` directory.
        """
        try:
            self._generate_and_run()
        finally:
            for aggregation in self.aggregations:
                aggregation.close()


class PredefinedExperiment(Experiment):
    """Representation of an experiment run for each provided LR system."""

    def __init__(
        self,
        name: str,
        data: DataStrategy,
        aggregations: Sequence[Aggregation],
        visualization_functions: list[Callable],
        output_path: Path,
        lrsystems: Iterable[LRSystem],
    ):
        super().__init__(name, data, aggregations, visualization_functions, output_path)
        self.lrsystems = lrsystems

    def _generate_and_run(self) -> None:
        for lrsystem in tqdm(self.lrsystems, desc=self.name, disable=not lir.is_interactive()):
            self._run_lrsystem(lrsystem)
