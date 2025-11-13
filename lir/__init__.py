import logging
import sys

from lir.transform import Transformer as Transformer  # as required by linting, later to be replaced by __all__


def is_interactive() -> bool:
    return logging.getLogger(__name__).level >= logging.WARNING and sys.stdout.isatty()
