from pathlib import Path

import confidence
import pytest

from lir.aggregation.metrics_bars import parse
from lir.config import ConfigValue, YamlParseError


@pytest.mark.parametrize(
    's,err',
    [
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
        (  # illegal argument: xrange
            """
        metrics:
          - cllr
        xrange: [0, 1]
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
