import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from tqdm import tqdm

import lir
from lir.aggregation import Aggregation, AggregationData
from lir.data.models import DataProvider, DataStrategy, concatenate_instances
from lir.lrsystems.lrsystems import LLRData, LRSystem


LOG = logging.getLogger(__name__)


class Experiment(ABC):
    """Representation of an experiment pipeline run for each provided LR system."""

    def __init__(
        self,
        name: str,
        data_provider: DataProvider,
        splitter: DataStrategy,
        outputs: Sequence[Aggregation],
        output_path: Path,
    ):
        self.name = name
        self.data_provider = data_provider
        self.splitter = splitter
        self.outputs = outputs
        self.output_path = output_path

    def _run_lrsystem(self, lrsystem: LRSystem, hyperparameters: dict[str, Any]) -> LLRData:
        """Run experiment on a single LR system configuration using the provided data(setup).

        First, the data is split into a training and testing subset, according to the provided
        data setup strategy. Next, the LR system is fitted using the training subset data and
        subsequently used to determine LLRs for the test subset data. The results are stored in
        a temporary list which contains the determined data of each test / train split.

        Metrics are collected as specified in the `metrics` mapping and returned after the experiment run.
        Visual representations of the obtained results for the specific LR system are generated as specified
        through the `visualization_functions` and stored in the `output_path` directory.
        """
        # Placeholders for numpy arrays of LLRs and labels obtained from each train/test split
        llr_sets: list[LLRData] = []

        # Split the data into a train / test subset, according to the provided DataStrategy. This could
        # for example be a simple binary split or a multiple fold cross validation split.
        for training_data, test_data in self.splitter.apply(self.data_provider.get_instances()):
            lrsystem.fit(training_data)
            subset_llr_results: LLRData = lrsystem.apply(test_data)

            # Store results (numpy arrays) into the placeholder lists
            llr_sets.append(subset_llr_results)

        # Combine collected numpy arrays after iteration over the train/test split(s)
        combined_llrs: LLRData = concatenate_instances(*llr_sets)

        # Collect and report results as configured by `outputs`
        results = AggregationData(llrdata=combined_llrs, lrsystem=lrsystem, parameters=hyperparameters)
        for output in self.outputs:
            output.report(results)

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
            for output in self.outputs:
                output.close()


class PredefinedExperiment(Experiment):
    """Representation of an experiment run for each provided LR system."""

    def __init__(
        self,
        name: str,
        data_provider: DataProvider,
        splitter: DataStrategy,
        outputs: Sequence[Aggregation],
        output_path: Path,
        lrsystems: Iterable[tuple[LRSystem, dict[str, Any]]],
    ):
        super().__init__(name, data_provider, splitter, outputs, output_path)
        self.lrsystems = lrsystems

    def _generate_and_run(self) -> None:
        for lrsystem, hyperparameters in tqdm(self.lrsystems, desc=self.name, disable=not lir.is_interactive()):
            self._run_lrsystem(lrsystem, hyperparameters)
