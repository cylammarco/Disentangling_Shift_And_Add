"""Command-line entry point."""

import argparse
from importlib.resources import files
from pathlib import Path

from .config import load_config
from .runner import run


def main(argv=None):
    args = _parser().parse_args(argv)
    action = write_example if args.init else _run_config
    action(_required_path(args))


def _required_path(args):
    path = args.init or args.config
    if path is None:
        raise SystemExit("Provide a YAML config or use --init PATH")
    return path


def _run_config(path):
    config = load_config(path)
    result, orbit = run(config)
    print(f"K1={orbit['K1']:.3f} K2={orbit['K2']:.3f} reduced chi2={result.chi2:.3f}")


def write_example(path):
    destination = Path(path)
    with destination.open("x") as target:
        target.write(files("saad").joinpath("example.yaml").read_text())


def _parser():
    parser = argparse.ArgumentParser(description="Binary shift-and-add disentangling")
    parser.add_argument("config", nargs="?", help="YAML configuration file")
    parser.add_argument("--init", metavar="PATH", help="Write example YAML configuration")
    return parser


if __name__ == "__main__":
    main()
