import socket

import pytest

from tests.conftest import (
    NetworkAccessWarning,
    _connect,
    _connect_ex,
)


def test_patched_connect_is_installed():
    assert socket.socket.connect is _connect
    assert socket.socket.connect_ex is _connect_ex


def test_connect_wrapper_warns_and_delegates(monkeypatch):
    calls = []

    def fake_connect(self, address):
        calls.append(address)
        raise OSError('offline')

    monkeypatch.setattr('tests.conftest._real_connect', fake_connect)
    sock = socket.socket()
    try:
        with (
            pytest.warns(NetworkAccessWarning, match='requires network access'),
            pytest.raises(OSError, match='offline'),
        ):
            socket.socket.connect(sock, ('example.com', 80))
    finally:
        sock.close()
    assert calls == [('example.com', 80)]


def test_connect_wrapper_warns_for_loopback(monkeypatch):
    # all addresses, including loopback ones, are treated as network access
    calls = []

    def fake_connect(self, address):
        calls.append(address)

    monkeypatch.setattr('tests.conftest._real_connect', fake_connect)
    sock = socket.socket()
    try:
        with pytest.warns(NetworkAccessWarning, match='requires network access'):
            socket.socket.connect(sock, ('127.0.0.1', 80))
    finally:
        sock.close()
    assert calls == [('127.0.0.1', 80)]


def test_connect_ex_wrapper_warns_and_delegates(monkeypatch):
    calls = []

    def fake_connect_ex(self, address):
        calls.append(address)
        return 0

    monkeypatch.setattr('tests.conftest._real_connect_ex', fake_connect_ex)
    sock = socket.socket()
    try:
        with pytest.warns(NetworkAccessWarning, match='requires network access'):
            result = socket.socket.connect_ex(sock, ('example.com', 443))
    finally:
        sock.close()
    assert calls == [('example.com', 443)]
    assert result == 0
