import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from tqdm import tqdm

import lir
from lir.aggregation import Aggregation, AggregationData
from lir.data.models import DataStrategy, concatenate_instances
from lir.lrsystems.lrsystems import LLRData, LRSystem


LOG = logging.getLogger(__name__)


class Experiment(ABC):
    """Representation of an experiment pipeline run for each provided LR system."""

    def __init__(
        self,
        name: str,
        data: DataStrategy,
        visualization_functions: list[Callable],
        outputs: Sequence[Aggregation],
        output_path: Path,
    ):
        self.name = name
        self.data = data
        self.visualization_functions = visualization_functions
        self.outputs = outputs
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

        # Generate output as configured by `outputs` and `visualization_functions`,
        # and write these output to the `output_path`.
        output_dir = self.output_path / lrsystem.name
        LOG.debug(f'writing visualizations to {output_dir}')
        for visualization_function in self.visualization_functions:
            visualization_function(output_dir, combined_llrs)

        # Construct a `results` dictionary of metrics indicating the performance of the given LR system
        LOG.debug(f'writing outputs to {output_dir}')
        for output in self.outputs:
            output.report(AggregationData(llrdata=combined_llrs, lrsystem=lrsystem, parameters=lrsystem.parameters))

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
        data: DataStrategy,
        visualization_functions: list[Callable],
        outputs: Sequence[Aggregation],
        output_path: Path,
        lrsystems: Iterable[LRSystem],
    ):
        super().__init__(name, data, visualization_functions, outputs, output_path)
        self.lrsystems = lrsystems

    def _generate_and_run(self) -> None:
        for lrsystem in tqdm(self.lrsystems, desc=self.name, disable=not lir.is_interactive()):
            self._run_lrsystem(lrsystem)
