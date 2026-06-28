import logging
import re
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np

from lir import Transformer
from lir.config.base import ConfigValue, config_parser, pop_field
from lir.data.models import InstanceData
from lir.util import check_type


LOG = logging.getLogger(__name__)


class SelectInstances(Transformer):
    """
    Select elements in a dataset from their indices.

    Parameters
    ----------
    select_element_fn : Callable[[int], bool]
        A function that takes a line number and returns True if it should be included, or False if it should be
        discarded.

    Examples
    --------
    This filter can be used in a YAML configuration:

    .. code-block:: yaml

        data:
          provider: [...]
          strategy: [...]
          filter:
            method: select_instances # drop instances unless their indices matches any of the following patterns
            indices:
              - 2                    # select element 2 -- the third record
              - 0-99                 # select element 0-99 (inclusive) -- the first 100 records
              - 0-99,800-899         # select element 0-99 and 800-899
              - /5                   # select every fifth element: 0, 5, 10, ...
    """

    def __init__(self, select_element_fn: Callable[[int], bool]):
        self._select_element_fn = np.vectorize(select_element_fn)

    def apply[DataType: InstanceData](self, instances: DataType) -> DataType:
        """
        Apply the selection to a dataset.

        Parameters
        ----------
        instances : InstanceData
            The dataset to select instances from.

        Returns
        -------
        InstanceData
            A dataset with only the selected instances.
        """
        selected = self._select_element_fn(np.arange(len(instances)))
        return instances[selected]


class _MatchIntPattern:
    """Collection of functions that parse strings into ranges."""

    @staticmethod
    def all(_: int) -> bool:  # numpydoc ignore=PR01,RT01
        """Return True always."""
        return True

    @staticmethod
    def any_of(functions: list[Callable[[int], bool]], n: int) -> bool:  # numpydoc ignore=PR01,RT01
        """Return True iff any of the conditions match."""
        for fn in functions:
            if fn(n):
                return True
        return False

    @staticmethod
    def equals(expected: int, n: int) -> bool:  # numpydoc ignore=PR01,RT01
        """Compare two `int` values and return True iff they are equal."""
        return n == expected

    @staticmethod
    def mod(mod: int, n: int) -> bool:  # numpydoc ignore=PR01,RT01
        """Return True iff the `int` parameter `n` is divisible by another `int` parameter `mod`."""
        return n % mod == 0

    @staticmethod
    def range(first: int, last: int, n: int) -> bool:  # numpydoc ignore=PR01,RT01
        """Return True iff the `int` parameter `n` is within a range."""
        return first <= n <= last

    @staticmethod
    def parse_str(spec: str) -> Callable[[int], bool]:  # numpydoc ignore=PR01,RT01
        """
        Parse a string parameter into a function.

        The string parameter specifies a condition. The returned function takes an `int` parameter and returns `True`
        iff the `int` value satisfied the condition.
        """
        spec = spec.strip()
        if spec.isnumeric():
            return partial(_MatchIntPattern.equals, int(spec))
        if re.match(r'^/\d+$', spec):
            return partial(_MatchIntPattern.mod, int(spec[1:]))
        if ',' in spec:
            return partial(_MatchIntPattern.any_of, [_MatchIntPattern.parse_str(item) for item in spec.split(',')])
        if re.match(r'^\d+-\d+$', spec):
            return partial(_MatchIntPattern.range, *[int(part) for part in spec.split('-')])
        raise ValueError(f'invalid pattern: {spec}')

    @staticmethod
    def parse(spec: None | list[str] | str) -> Callable[[int], bool]:  # numpydoc ignore=PR01,RT01
        """
        Parse a parameter `spec` into a function.

        The returned function takes an `int` parameter and returns `True` iff it satisfies the condition as specified
        by `spec`.

        If `spec` is None, the returned function always returns `True`.

        If `spec` is a `str`, the returned function returns `True` iff its `int` parameter satisfies the condition in
        `spec`.

        If `spec` is a list of `str`, the returned function `True` iff its `int` parameter satisfies the condition in
        any of the elements of `spec`.
        """
        match spec:
            case None:
                return _MatchIntPattern.all
            case str():
                return _MatchIntPattern.parse_str(spec)
            case list():
                return partial(_MatchIntPattern.any_of, [_MatchIntPattern.parse(item) for item in spec])
            case _:
                raise ValueError(f'unknown pattern type: {type(spec)}; the value was: {spec}')


@config_parser
def parse_select_instances(config: ConfigValue, _: Path) -> SelectInstances:  # numpydoc ignore=PR01,RT01
    """Parse SelectInstances configuration."""
    select_rows_cfg = pop_field(config, 'indices', validate=partial(check_type, list))
    return SelectInstances(_MatchIntPattern.parse(select_rows_cfg))
