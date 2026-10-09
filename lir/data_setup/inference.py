from collections.abc import Iterable
from pathlib import Path

from lir import DataProvider, DataSetup, InstanceData
from lir.config import ConfigAttribute, ConfigValue, config_parser
from lir.config.data import parse_data_provider


class InferenceDataSetup(DataSetup):
    """
    A data setup that uses two separate data providers for training and inference.

    If no data provider for training is specified, the training phase is skipped and the model is assumed to be fitted.

    If no data provider for inference is specified, the model will be trained only, and the inference phase is skipped.

    Parameters
    ----------
    training : DataProvider, optional
        The :class:`~lir.DataProvider` that loads the training data.
    inference : DataStrategy, optional
        The :class:`~lir.DataProvider` that determines how the data are used.
    """

    def __init__(self, training: DataProvider | None, inference: DataProvider | None):
        self.training = training
        self.inference = inference
        if self.training is None and self.inference is None:
            raise ValueError('specify at least one of training data and inference data')

    def get_train_inference_pairs(self) -> Iterable[tuple[InstanceData | None, InstanceData | None]]:
        """
        Return the data in the form of one or more train/test splits.

        This method follows three steps:
        - retrieve instances from the data provider;
        - pass them through the filter by calling its `apply()` method;
        - apply the data strategy to arrange them into one or more train/test splits.

        This method caches results for speeding up consecutive calls if the class is instantiated with
        ``use_cache=True``.

        Returns
        -------
        Iterable[tuple[InstanceData, InstanceData]]
            An iterator over tuples of train/test splits.
        """
        training_data = self.training.get_instances() if self.training is not None else None
        inference_data = self.inference.get_instances() if self.inference is not None else None
        return [(training_data, inference_data)]


@config_parser(
    attributes=[
        ConfigAttribute('training', DataProvider, required=False),
        ConfigAttribute('inference', DataProvider, required=False),
    ]
)
def parse(config: ConfigValue, output_path: Path) -> InferenceDataSetup:
    """
    Parse a data setup with a separate training and inference dataset.

    Parameters
    ----------
    config : ConfigValue
        Configuration section containing provider and split strategy.
    output_path : Path
        Output path for created objects.

    Returns
    -------
    InferenceDataSetup
        Parsed data setup.
    """
    with config:
        training_data_config = config.pop('training', required=False)
        training_data = (
            parse_data_provider(training_data_config, output_path) if training_data_config.value is not None else None
        )

        inference_data_config = config.pop('inference', required=False)
        inference_data = (
            parse_data_provider(inference_data_config, output_path) if inference_data_config.value is not None else None
        )

        return InferenceDataSetup(training_data, inference_data)
