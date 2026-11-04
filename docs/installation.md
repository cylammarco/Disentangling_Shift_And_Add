# Installation

SAAD requires Python 3.10 or newer. The package declares NumPy 1.26 and 2.x compatible dependencies. Continuous integration tests Python 3.10 with NumPy 1.26, Python 3.11 with NumPy 2.0, and Python 3.12 with a current NumPy 2 release.

## From a checkout

```bash
git clone https://github.com/cylammarco/Disentangling_Shift_And_Add.git
cd Disentangling_Shift_And_Add
python -m pip install .
saad --help
```

For development, install the source tree in editable mode:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

The distribution name is `saad-disentangling`; the import name and command are both `saad`. The repository can also be installed directly with `python -m pip install git+https://github.com/cylammarco/Disentangling_Shift_And_Add.git`.

## Build these docs locally

```bash
python -m pip install -e '.[docs]'
sphinx-build -b html -n -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html`. The repository's `.readthedocs.yaml` installs the same `docs` extra and builds `docs/conf.py` on Read the Docs. The figures are committed with the documentation, so hosted builds do not run the numerical example. To regenerate them from `example/make_spectra_SB2.py`, run `python docs/generate_figures.py` from the repository root.

## First run

The fastest end-to-end route uses the bundled synthetic binary spectra and a short configuration in `example/quickstart.yaml`. Follow the [quickstart](quickstart.md). For your own spectra, read the [data and configuration reference](configuration.md) before running the CLI.
