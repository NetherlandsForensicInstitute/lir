from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import confidence

from lir.aggregation import Aggregation, AggregationData
from lir.config.base import ContextAwareDict
from lir.config.lrsystem_architectures import parse_lrsystem
from lir.config.util import simplify_data_structure
from lir.data.models import InstanceData, LLRData, concatenate_instances
from lir.lrsystems.lrsystems import LRSystem


class Experiment(ABC):
    """Representation of an experiment pipeline run for each provided LR system."""

    def __init__(
        self,
        name: str,
        outputs: Sequence[Aggregation],
        output_path: Path,
    ):
        self.name = name
        self.outputs = outputs
        self.output_path = output_path

    def _run_lrsystem(
        self,
        lrsystem_config: ContextAwareDict,
        split_data: Iterable[tuple[InstanceData, InstanceData]],
        parameters: dict[str, Any],
        run_name: str,
        data_config: ContextAwareDict,
    ) -> LLRData:
        """Run experiment on a single LR system configuration using the provided data.

        The LR system is fitted using the training subset data and
        subsequently used to determine LLRs for the test subset data. The results are stored in
        a temporary list which contains the determined data of each test / train split.

        The collected results are combined and passed to the configured `outputs` aggregations,
        which may write metrics and visualizations to the `output_path` directory. The combined
        LLR data is returned.

        Next to this, the configuration of both the data and LR system are stored in the output
        directory for future reference.
        """
        run_output_dir = self.output_path / run_name
        run_output_dir.mkdir(exist_ok=True, parents=True)

        lrsystem = parse_lrsystem(deepcopy(lrsystem_config), run_output_dir)

        # Turn the configurations into dictionaries for writing to YAML files.
        lrsystem_config_dict = simplify_data_structure(lrsystem_config)
        data_config_dict = simplify_data_structure(data_config)

        # Check that simplify_data_structure returned a dict for type checking.
        if not isinstance(lrsystem_config_dict, dict) or not isinstance(data_config_dict, dict):
            raise ValueError('hyperparameters are not the expected type (dict)')

        confidence.dumpf(confidence.Configuration(lrsystem_config_dict), run_output_dir / 'lrsystem.yaml')
        confidence.dumpf(confidence.Configuration(data_config_dict), run_output_dir / 'data.yaml')

        # Convert the split_data iterable to a list to allow multiple iterations over the splits.
        # E.g. one iteration for validation and one for case LLR generation.
        split_data_list = list(split_data)
        if not split_data_list:
            raise ValueError('data splitting strategy did not produce any train/test splits')

        # Placeholders for numpy arrays of LLRs and labels obtained from each train/test split
        llr_sets: list[LLRData] = []

        for training_data, test_data in split_data_list:
            lrsystem.fit(training_data)
            subset_llr_results: LLRData = lrsystem.apply(test_data)

            # Store results (numpy arrays) into the placeholder lists
            llr_sets.append(subset_llr_results)

        # Combine collected numpy arrays after iteration over the train/test split(s)
        combined_llrs: LLRData = concatenate_instances(*llr_sets)

        # Create a lazy factory for full-data-fitted model with memoization
        _cached_full_fit_lrsystem = None

        def get_full_fit_lrsystem() -> LRSystem:
            nonlocal _cached_full_fit_lrsystem
            if _cached_full_fit_lrsystem is None:
                first_training_data, first_test_data = split_data_list[0]
                full_training_data = concatenate_instances(first_training_data, first_test_data)
                _cached_full_fit_lrsystem = parse_lrsystem(deepcopy(lrsystem_config), run_output_dir)
                _cached_full_fit_lrsystem.fit(full_training_data)
            return _cached_full_fit_lrsystem

        # Collect and report results as configured by `outputs`
        results = AggregationData(
            llrdata=combined_llrs,
            lrsystem=lrsystem,
            parameters=parameters,
            run_name=run_name,
            get_full_fit_lrsystem=get_full_fit_lrsystem,
        )
        for output in self.outputs:
            output.report(results)

        return combined_llrs

    @abstractmethod
    def _generate_and_run(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        """Run experiment the configured experiment(s).

        This method ensures that all outputs are properly closed after the experiment run.
        """
        try:
            self._generate_and_run()
        finally:
            for output in self.outputs:
                output.close()
