#!/usr/bin/env python3

import argparse
import datetime
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import confidence

from lir import registry
from lir.config.base import YamlParseError, pop_field
from lir.config.experiment_strategies import parse_experiments_setup
from lir.config.substitution import _expand

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

    logging.getLogger("").setLevel(logging.DEBUG)


def initialize_logfile(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / "log.txt")
    fh.setFormatter(logging.Formatter("[%(asctime)-15s %(levelname)s] %(name)s: %(message)s"))
    fh.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(fh)


def error(msg: str) -> None:
    sys.stderr.write(f"{msg}\n")
    if LOG.level <= logging.DEBUG:
        raise
    sys.exit(1)


def load_configuration(path: str) -> dict[str, Any]:
    cfg = confidence.loadf(path)
    cfg = confidence.Configuration(
        cfg, {"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")}
    )

    return _expand(cfg)


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

    cfg = load_configuration(args.setup)

    output_dir = pop_field([], cfg, "output", validate=Path)
    initialize_logfile(output_dir)

    try:
        experiments = parse_experiments_setup(cfg, output_dir)
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
