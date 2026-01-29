import logging
from collections.abc import Callable
from typing import Any, Self

from lir import Transformer
from lir.data.models import InstanceData, concatenate_instances, get_instances_by_category


LOG = logging.getLogger(__name__)


class CategoricalCompositeTransformer(Transformer):
    """
    Composite transformer.

    Incoming data is categorized by a category field. For each category, a separate transformer is used.
    """

    def __init__(self, factory: Callable[[], Transformer], category_field: str):
        """
        Initialize the composite transformer.

        :param factory: a callable that takes a category field and returns a transformer.
        :param category_field: the category field to use
        """
        self.factory = factory
        self.category_field = category_field
        self._transformers: dict[str, Transformer] = {}
        self._category_shape: tuple[int] | None = None

    def _get_transformer(self, value: Any, create_if_missing: bool) -> Transformer:
        value = str(value)
        if value not in self._transformers:
            if create_if_missing:
                self._transformers[value] = self.factory()
            else:
                raise ValueError(f'no fit for value: {value}')

        return self._transformers[value]

    def fit(self, instances: InstanceData) -> Self:
        """
        Fit the transformer for all categories found in `instances`.

        :param instances: a set of instances
        """
        # reset
        self._transformers = {}
        self._category_shape = None

        # fit the models
        for category, subset in get_instances_by_category(instances, self.category_field):
            LOG.debug(f'fitting sub model for category: {category}')
            self._category_shape = category.shape
            self._get_transformer(category, True).fit(subset)

        return self

    def apply(self, instances: InstanceData) -> InstanceData:
        """
        Apply the specialized transformers for all instances in `instances`, based on their categories.

        :param instances: a set of instances
        :return: the transformed instances
        """
        if self._category_shape is None:
            raise ValueError('apply() called before fit()')

        result: list[InstanceData] = []
        for category, subset in get_instances_by_category(instances, self.category_field, self._category_shape):
            LOG.debug(f'applying sub model for category: {category}')
            result.append(self._get_transformer(category, False).apply(subset))

        LOG.debug('concatenating results')
        return concatenate_instances(*result)
