import logging
import sys

from lir.transform import Transformer as Transformer


def is_interactive() -> bool:
    return logging.getLogger(__name__).level >= logging.WARNING and sys.stdout.isatty()
