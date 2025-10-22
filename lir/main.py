#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

import confidence

from lir import registry
from lir.config.base import YamlParseError
from lir.config.experiment_strategies import parse_experiments_setup

LOG = logging.getLogger(__file__)
DEFAULT_LOGLEVEL = logging.WARNING


def setup_logging(file_path: str, level_increase: int) -> None:
    """
    Setup logging to stderr and to a file.

    :param file_path: target file
    :param level_increase: log level for stderr, relative to the default log level
    """
    loglevel = max(
        logging.DEBUG, min(logging.CRITICAL, DEFAULT_LOGLEVEL - level_increase * 10)
    )

    # setup formatter
    log_format = "[%(asctime)-15s %(levelname)s] %(name)s: %(message)s"
    fmt = logging.Formatter(log_format)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(loglevel)
    logging.getLogger().addHandler(ch)

    # setup a file handler
    fh = RotatingFileHandler(
        file_path, maxBytes=10 * 1024 * 1024, backupCount=2, delay=True
    )
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logging.getLogger().addHandler(fh)
    if os.path.exists(file_path):
        fh.doRollover()

    logging.getLogger("").setLevel(logging.INFO)
    logging.getLogger("lir").setLevel(loglevel)


def error(msg: str) -> None:
    sys.stderr.write(f"{msg}\n")
    if LOG.level <= logging.DEBUG:
        raise
    sys.exit(1)


def main() -> None:
    app_name = "benchmark"

    parser = argparse.ArgumentParser(
        description="Run all or some of the parts of project"
    )

    parser.add_argument(
        "--setup",
        metavar="PATH",
        help="path to YAML file describing the evaluation setup",
        required=True,
    )
    parser.add_argument(
        "--experiment",
        help="run a specific experiment (defaults to all experiments)",
        nargs="*",
    )
    parser.add_argument(
        "--list-experiments",
        help="prints a list of configured experiments",
        action="store_true",
    )
    parser.add_argument(
        "--list-registry",
        help="prints a list of registered components",
        action="store_true",
    )

    parser.add_argument("-v", help="increases verbosity", action="count", default=0)
    parser.add_argument("-q", help="decreases verbosity", action="count", default=0)
    args = parser.parse_args()

    setup_logging(f"{app_name}.log", args.v - args.q)

    if args.list_registry:
        for name in registry.registry():
            print(name)
        return

    try:
        experiments = parse_experiments_setup(confidence.loadf(args.setup))
    except YamlParseError as e:
        error(f"error while parsing {args.setup}: {str(e)}")
        raise  # this statement is not reachable, but helps code validation

    if args.list_experiments:
        for name, experiment in experiments.items():
            print(name)
        return

    if args.experiment:
        for name in args.experiment:
            if name in experiments:
                experiments[name].run()
            else:
                error(f"no such experiment: {name}")
    else:
        for experiment in experiments.values():
            experiment.run()


if __name__ == "__main__":
    main()
