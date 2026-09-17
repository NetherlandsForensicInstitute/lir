from pathlib import Path

import confidence
import pytest

from lir.aggregation.metrics_bars import parse
from lir.config import ConfigValue, YamlParseError


@pytest.mark.parametrize(
    's,err',
    [
        # plot_params and metrics are optional; check if they can be omitted.
        ('', None),
        # plot_params is optional; check if it can be omitted.
        (
            """
            metrics:
              - cllr
              - cllr_min
              """,
            None,
        ),
        # valid configuration with plot_params and metrics.
        (
            """
            metrics:
              - cllr
              - cllr_min
            plot_params:
              ylim: [0, 1]
              xlabel: run
            """,
            None,
        ),
        # invalid configuration: xlabel should be an argument to plot_params, not a top-level key.
        (
            """
            metrics:
              - cllr
            xlabel: run
            """,
            YamlParseError,
        ),
    ],
)
def test_parse(s: str, err: type[BaseException] | None):
    cfg = ConfigValue.wrap([], confidence.loads(s))
    if err:
        with pytest.raises(err):
            parse().parse(cfg, Path('/'))
    else:
        parse().parse(cfg, Path('/'))
