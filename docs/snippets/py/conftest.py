import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

from lir import LLRData
from lir.aggregation import AggregationData


@pytest.fixture
def aggregation_data() -> Iterable[list[AggregationData]]:
    with tempfile.TemporaryDirectory() as tmpdir:
        data = []
        for run_name, llrs, hypothesis in [
            ('run1', np.array([-1, 0, 1]), np.array([0, 1, 1])),
            ('run2', np.array([-2, -1, 0]), np.array([0, 1, 1])),
            ('run3', np.array([0, 2.5, 5]), np.array([0, 1, 1])),
        ]:
            data.append(
                AggregationData(
                    llrdata=LLRData(features=llrs, hypothesis=hypothesis, average_llr=np.average(llrs)),
                    lrsystem=None,
                    parameters={},
                    run_name=run_name,
                    experiment_output_dir=Path(tmpdir),
                    run_output_dir=Path(tmpdir) / run_name,
                )
            )

        yield data
