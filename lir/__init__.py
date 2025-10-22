import logging
import sys


def is_interactive() -> bool:
    return logging.getLogger(__name__).level >= logging.WARNING and sys.stdout.isatty()
