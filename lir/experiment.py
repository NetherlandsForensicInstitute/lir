import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Iterable, Callable, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

import lir
from lir.aggregation import Aggregation
from lir.data.models import DataStrategy
from lir.lrsystems.lrsystems import LRSystem


LOG = logging.getLogger(__name__)


class Experiment(ABC):
    """Representation of an experiment pipeline run for each provided LR system."""

    def __init__(
        self,
        name: str,
        data: DataStrategy,
        aggregations: Sequence[Aggregation],
        visualization_functions: List[Callable],
        output_path: Path,
    ):
        self.name = name
        self.data = data
        self.aggregations = aggregations
        self.visualization_functions = visualization_functions
        self.output_path = output_path

    def _run_lrsystem(
        self, lrsystem: LRSystem
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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
        llrs = []
        labels: List[np.ndarray] = []

        # Split the data into a train / test subset, according to the provided DataSetup. This could
        # for example be a simple binary split or a multiple fold cross validation split.
        for (features_train, labels_train, meta_train), (
            features_test,
            labels_test,
            meta_test,
        ) in self.data:
            lrsystem.fit(features_train, labels_train, meta_train)
            subset_llrs, subset_labels, subset_meta = lrsystem.apply(
                features_test, labels_test, meta_test
            )
            # Store results (numpy arrays) into the placeholder lists
            llrs.append(subset_llrs)
            if subset_labels is not None:
                labels.append(subset_labels)

        # Combine collected numpy array's after iteration over the train/test split(s)
        llrs = np.concatenate(llrs)
        labels: Optional[np.ndarray] = np.concatenate(labels) if labels else None

        # Generate visualization output as configured by `visualization_functions`
        # and write graphical output to the `output_path`.
        output_dir = self.output_path / lrsystem.name
        LOG.debug(f"writing visualizations to {output_dir}")
        for visualization_function in self.visualization_functions:
            visualization_function(output_dir, llrs, labels)

        # Construct a `results` dictionary of metrics indicating the performance of the given LR system
        for aggregation in self.aggregations:
            aggregation.report(llrs, labels, lrsystem.parameters)

        return llrs, labels

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
        visualization_functions: List[Callable],
        output_path: Path,
        lrsystems: Iterable[LRSystem],
    ):
        super().__init__(name, data, aggregations, visualization_functions, output_path)
        self.lrsystems = lrsystems

    def _generate_and_run(self) -> None:
        for lrsystem in tqdm(
            self.lrsystems, desc=self.name, disable=not lir.is_interactive()
        ):
            self._run_lrsystem(lrsystem)
