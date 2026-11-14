import socket
import warnings

import matplotlib
import pytest

from lir.algorithms.bootstraps import BootstrapAtData
from lir.algorithms.logistic_regression import LogitCalibrator
from lir.data.models import DataProvider, FeatureData, InstanceData, LLRData
from lir.datasets.synthesized_normal_binary import SynthesizedNormalBinaryData, SynthesizedNormalData
from lir.transform import as_transformer


# tests only save figures to file; the non-interactive Agg backend is considerably faster than GUI backends
matplotlib.use('Agg')


class NetworkAccessWarning(UserWarning):
    """Warning that a unit test attempted to access the network."""


# unit tests should run without network access (issue #520): warn when a socket tries to connect
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def _connect(self, address):  # type: ignore[no-untyped-def]
    warnings.warn(
        f'unit test should run offline, but requires network access (connecting to {address!r})',
        NetworkAccessWarning,
        stacklevel=2,  # point at the code that initiated the connection
    )
    return _real_connect(self, address)


def _connect_ex(self, address):  # type: ignore[no-untyped-def]
    warnings.warn(
        f'unit test should run offline, but requires network access (connecting to {address!r})',
        NetworkAccessWarning,
        stacklevel=2,  # point at the code that initiated the connection
    )
    return _real_connect_ex(self, address)


socket.socket.connect = _connect
socket.socket.connect_ex = _connect_ex


@pytest.fixture(autouse=True)
def _warn_on_network_access():
    """Keep NetworkAccessWarning a warning, even if the tests are run with `-W error`."""
    # 'default' filters repeated warnings for the same address out (issue #520)
    with warnings.catch_warnings():
        warnings.simplefilter('default', category=NetworkAccessWarning)
        yield


@pytest.fixture
def synthesized_normal_data_provider() -> DataProvider:
    h1_data = SynthesizedNormalData(mean=0, std=1, size=100)  # H1
    h2_data = SynthesizedNormalData(mean=2, std=1, size=100)  # H2
    return SynthesizedNormalBinaryData(h1_data, h2_data, seed=42)


@pytest.fixture
def synthesized_normal_data(synthesized_normal_data_provider: DataProvider) -> InstanceData:
    return synthesized_normal_data_provider.get_instances()


@pytest.fixture
def synthesized_llrs(synthesized_normal_data: FeatureData) -> LLRData:
    return as_transformer(LogitCalibrator()).fit_apply(synthesized_normal_data).replace_as(LLRData)


@pytest.fixture
def synthesized_llrs_with_interval(synthesized_normal_data: FeatureData) -> LLRData:
    bootstrap = BootstrapAtData(steps=[('clf', LogitCalibrator())])
    return bootstrap.fit_apply(synthesized_normal_data)
