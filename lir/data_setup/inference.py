from collections.abc import Iterable
from pathlib import Path

from lir import DataProvider, DataSetup, InstanceData, Transformer
from lir.config import ConfigAttribute, ConfigValue, config_parser
from lir.config.data import parse_data_provider
from lir.config.transform import parse_module
from lir.transform import Identity


class InferenceDataSetup(DataSetup):
    """
    A data setup that uses two separate data providers for training and inference.

    If no data provider for training is specified, the training phase is skipped and the model is assumed to be fitted.

    If no data provider for inference is specified, the model will be trained only, and the inference phase is skipped.

    Parameters
    ----------
    training : DataProvider, optional
        The :class:`~lir.DataProvider` that loads the training data.
    inference : DataProvider, optional
        The :class:`~lir.DataProvider` that loads the inference data.
    data_filter : Transformer | None, optional
        An optional filter (:class:`~lir.Transformer`) that is applied to both the training and the inference data,
        after they are retrieved from their data providers. The filter's `apply()` method is called without `fit()`,
        and it may change the number of instances.
    """

    def __init__(
        self,
        training: DataProvider | None,
        inference: DataProvider | None,
        data_filter: Transformer | None = None,
    ):
        self.training = training
        self.inference = inference
        self.filter = data_filter or Identity()
        if self.training is None and self.inference is None:
            raise ValueError('specify at least one of training data and inference data')
        self._pairs: tuple[InstanceData | None, InstanceData | None] | None = None

    def get_train_inference_pairs(self) -> Iterable[tuple[InstanceData | None, InstanceData | None]]:
        """
        Return the training and inference data as a single (training, inference) pair.

        The instances of the data providers are fetched once, passed through the filter, and cached, so consecutive
        calls return the same data.

        Returns
        -------
        Iterable[tuple[InstanceData | None, InstanceData | None]]
            An iterator over a single tuple of (training data, inference data), where an element is
            `None` if the corresponding data provider is not configured.
        """
        if self._pairs is None:
            training_data = self._load(self.training)
            inference_data = self._load(self.inference)
            self._pairs = (training_data, inference_data)
        return (self._pairs,)

    def _load(self, provider: DataProvider | None) -> InstanceData | None:
        """
        Retrieve the instances from a provider and apply the filter.

        Parameters
        ----------
        provider : DataProvider | None
            The data provider to retrieve the instances from, or `None` if no provider is configured.

        Returns
        -------
        InstanceData | None
            The filtered instances, or `None` if no provider is configured.
        """
        if provider is None:
            return None
        return self.filter.apply(provider.get_instances())


@config_parser(
    attributes=[
        ConfigAttribute('training', DataProvider, required=False),
        ConfigAttribute('inference', DataProvider, required=False),
        ConfigAttribute('filter', Transformer, required=False),
    ]
)
def parse(config: ConfigValue, output_path: Path) -> InferenceDataSetup:
    """
    Parse a data setup with a separate training and inference dataset.

    Parameters
    ----------
    config : ConfigValue
        Configuration section containing the training and inference data providers and an optional filter.
    output_path : Path
        Output path for created objects.

    Returns
    -------
    InferenceDataSetup
        Parsed data setup.
    """
    with config:
        data_providers: dict[str, DataProvider | None] = {}
        for field in ('training', 'inference'):
            field_config = config.pop(field, required=False)
            data_providers[field] = (
                parse_data_provider(field_config, output_path) if field_config.value is not None else None
            )

        data_filter = parse_module(config.pop('filter', required=False), output_path)

        return InferenceDataSetup(data_providers['training'], data_providers['inference'], data_filter)
