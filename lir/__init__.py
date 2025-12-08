import sys

from lir.transform import Transformer as Transformer  # as required by linting, later to be replaced by __all__


def is_interactive() -> bool:
    return sys.stdout.isatty()
