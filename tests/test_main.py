import argparse
import copy
import importlib
import sys
from pathlib import Path

import pytest

from lir.main import main

from . import resources as resources_package


def raise_on_exit(exitval: int):
    raise ValueError(f"exit: {exitval}")


config = importlib.resources.files(resources_package) / 'setup.yaml'
sys.exit = raise_on_exit


@pytest.mark.parametrize("command", [
    "--list-registry",
    f"--list-experiments {config}",
    f"{config}",
    f"--experiment exp1 {config}",
    f"--experiment exp1 --experimen exp2 {config}",
])
def test_main_commands(command: str):
    try:
        main(command.split())
    except Exception:
        pytest.fail('Unexpected failure while parsing valid CLI arguments')


@pytest.mark.parametrize("command", [
    "--list-regstry",
    f"--experiment exp3 {config}",
    f"{config}x",
])
def test_invalid_main_commands(command: str):
    with pytest.raises(Exception):
        main(command.split())
