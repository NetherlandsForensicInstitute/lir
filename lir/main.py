#!/usr/bin/env python3

import argparse
import logging
import sys
from pathlib import Path

import confidence

from lir import registry
from lir.config.base import YamlParseError
from lir.config.experiment_strategies import parse_experiments_setup


LOG = logging.getLogger(__name__)
DEFAULT_LOGLEVEL = logging.WARNING


def setup_logging(file_path: str, level_increase: int) -> None:
    """
    Setup logging to stderr and to a file.

    :param file_path: target file
    :param level_increase: log level for stderr, relative to the default log level
    """
    loglevel = max(logging.DEBUG, min(logging.CRITICAL, DEFAULT_LOGLEVEL - level_increase * 10))

    # setup formatter
    log_format = '[%(asctime)-15s %(levelname)s] %(name)s: %(message)s'
    fmt = logging.Formatter(log_format)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(loglevel)
    logging.getLogger().addHandler(ch)

    logging.getLogger().setLevel(logging.DEBUG)


def initialize_logfile(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / 'log.txt')
    fh.setFormatter(logging.Formatter('[%(asctime)-15s %(levelname)s] %(name)s: %(message)s'))
    fh.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(fh)


def error(msg: str, e: Exception | None = None) -> None:
    sys.stderr.write(f'{msg}\n')
    if e and LOG.level <= logging.DEBUG:
        raise e
    sys.exit(1)


def main(args: list[str] | None = None) -> None:
    app_name = 'lir'

    parser = argparse.ArgumentParser(description='Run all or some of the parts of project')

    parser.add_argument(
        'setup',
        metavar='FILENAME',
        help='path to YAML file describing the evaluation setup',
        nargs='?',
    )
    parser.add_argument(
        '--experiment',
        help='run a specific experiment (defaults to all experiments)',
        action='append',
    )
    parser.add_argument(
        '--list-experiments',
        help='prints a list of configured experiments',
        action='store_true',
    )
    parser.add_argument(
        '--list-registry',
        help='prints a list of registered components',
        action='store_true',
    )

    parser.add_argument('-v', help='increases verbosity', action='count', default=0)
    parser.add_argument('-q', help='decreases verbosity', action='count', default=0)
    args = parser.parse_args(args)

    setup_logging(f'{app_name}.log', args.v - args.q)

    if args.list_registry:
        for name in registry.registry():
            print(name)
        return

    ### an experiment setup is required beyond this point ###

    if not args.setup:
        parser.error('missing FILENAME argument')

    # Add directories (setup file and current folder) to sys.path for custom components.
    sys.path.append(str(Path(args.setup).resolve().parent))
    LOG.debug(f'added {Path(args.setup).resolve().parent} to sys.path')

    sys.path.append(str(Path().resolve()))
    LOG.debug(f'added {Path().resolve()} to sys.path')

    try:
        experiments, output_dir = parse_experiments_setup(confidence.loadf(args.setup))
    except YamlParseError as e:
        error(f'error while parsing {args.setup}: {str(e)}', e)
        raise  # this statement is not reachable, but helps code validation

    initialize_logfile(output_dir)

    if args.list_experiments:
        for name, _experiment in experiments.items():
            print(name)
        return

    if args.experiment:
        for name in args.experiment:
            if name not in experiments:
                error(f'no such experiment: {name}')

        for name in set(args.experiment):
            experiments[name].run()
    else:
        for experiment in experiments.values():
            experiment.run()


if __name__ == '__main__':
    main()
