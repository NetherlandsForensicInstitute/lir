#!/usr/bin/env python3

import argparse
import datetime
import logging
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

import confidence
from joblib import Parallel, delayed

from lir import registry
from lir.config.base import YamlParseError, _expand, pop_field
from lir.config.experiment_strategies import parse_experiments
from lir.experiment import Experiment


LOG = logging.getLogger(__name__)
DEFAULT_LOGLEVEL = logging.WARNING


def setup_logging(level_increase: int) -> None:
    """Set up logging to stderr and to a file.

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
    """Set up logfile for debugging purposes when running experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / 'log.txt')
    fh.setFormatter(logging.Formatter('[%(asctime)-15s %(levelname)s] %(name)s: %(message)s'))
    fh.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(fh)


def copy_yaml_definition(output_dir: Path, config_yaml_path: Path) -> None:
    """Copy the YAML definition for a given LR system experiment to persist the used configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_yaml_path, output_dir / 'config.yaml')


def initialize_experiments(
    cfg: confidence.Configuration,
) -> tuple[Mapping[str, Experiment], Path]:
    """Extract which Experiment to run as dictated in the configuration.

    The following pre-defined variables are injected to the configuration:

    - `timestamp`: a formatted timestamp of the current date/time

    :param cfg: a `Configuration` object describing the experiments
    :return: a tuple with two elements: (1) mapping of names to experiments; (2) path to output directory
    """
    cfg = confidence.Configuration(cfg, {'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')})  # noqa: DTZ005

    cfg = _expand([], cfg)

    output_dir = pop_field(cfg, 'output_path', validate=Path)
    initialize_logfile(output_dir)

    return parse_experiments(cfg, output_dir), output_dir


def error(msg: str, e: Exception | None = None) -> None:
    """Stop execution with given error message or raise exception."""
    sys.stderr.write(f'{msg}{" (see log for details)" if e else ""}\n')
    LOG.debug(f'abort: {msg}', exc_info=e)
    sys.exit(1)


def main(input_args: list[str] | None = None) -> None:
    """Provide Command Line Interface (CLI) to LiR."""
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
    parser.add_argument(
        '--n-jobs',
        help='Enable parallel execution of experiments. Use 0 or 1 to disable parallelism, or -1 to use all available'
        ' cores, -2 to use all but one core, etc. The parallelism is at the experiment level, so each experiment will'
        ' be run in its own process. For more information, see the joblib documentation:'
        ' https://joblib.readthedocs.io/en/latest/parallel.html',
        type=int,
        default=0,
    )

    parser.add_argument('-v', help='increases verbosity', action='count', default=0)
    parser.add_argument('-q', help='decreases verbosity', action='count', default=0)
    args = parser.parse_args(input_args)

    setup_logging(args.v - args.q)

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
        experiments, output_dir = initialize_experiments(confidence.loadf(args.setup))
    except YamlParseError as e:
        error(f'error while parsing {args.setup}: {str(e)}', e)
        raise  # this statement is not reachable, but helps code validation

    copy_yaml_definition(output_dir, Path(args.setup))

    if args.list_experiments:
        for name in experiments:
            print(name)
        return

    if args.experiment:
        for name in args.experiment:
            if name not in experiments:
                error(f'no such experiment: {name}')

        for name in set(args.experiment):
            experiments[name].run()
    else:
        n_jobs = args.n_jobs
        if n_jobs not in (0, 1):
            # Run in parallel using joblib. Whilst joblib can handle 1 job, it is more efficient to run sequentially.
            LOG.info(f'Running selected experiments in parallel using {n_jobs} cores.')
            Parallel(n_jobs=n_jobs)(delayed(experiment.run)() for experiment in experiments.values())
        else:
            # Run sequentially.
            LOG.info('Running selected experiments sequentially.')
            for experiment in experiments.values():
                experiment.run()


if __name__ == '__main__':
    main()
