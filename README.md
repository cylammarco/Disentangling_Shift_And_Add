# SAAD

This fork packages shift-and-add spectral disentangling as `saad`. The installable workflow supports binary systems, optional stationary nebular emission, K1/K2 grid search, and diagnostic plots. Tomer Shenar's upstream V2.0 workflow remains under `example/legacy/` for binaries and triples.

Full installation, quickstart, method, configuration, outputs, and Python API documentation: [docs/index.md](docs/index.md).

## Install

From a clone of this repository:

```bash
python -m pip install .
# or
uv pip install .
```

For development, use `python -m pip install -e '.[test]'` or `uv pip install -e '.[test]'`. A direct GitHub install uses `python -m pip install git+https://github.com/cylammarco/Disentangling_Shift_And_Add.git`. Legacy `python setup.py install` also works. The future PyPI distribution name is `saad-disentangling`; import name and command remain `saad`. The name `saad` is already used by an unrelated PyPI project.

Dependencies support NumPy 1.26 and 2.x. CI tests both major versions.

## Run binary workflow

Create and edit a configuration file, then run:

```bash
saad --init config.yaml
saad config.yaml
# or
python -m saad.cli config.yaml
```

Paths in YAML resolve relative to the YAML file. `obs_path` contains spectra. For `obs_format: TXT`, add `ObsDat.txt` with `MJD obsname` columns and one two-column spectrum per row. For `obs_format: FITS`, observation times come from the header key in `mjd_header_keyword`. Set `ranges` to wavelength pairs for chi-square scoring, and set `grid_dis: false` to use orbital K1/K2 values directly. `inter_kind` accepts `linear`, `cubic`, or flux-conserving `resample`.

Results go to `output_path`: scaled primary and secondary spectra, `fit_summary.json`, optional K1/K2 grid, and plots selected by `plot_extremes`, `plot_fits`, `plot_conv`, and `plot_itr`. With `neb_lines: true`, a stationary nebular spectrum is also fitted and saved. All spectra must use the same wavelength units as `ranges`.

Python callers can use `from saad import load_config, run` and `result, orbit = run(load_config("config.yaml"))`. Run tests with `python -m pytest`.

## Examples and legacy triples

`example/make_spectra_SB2.py` and `example/make_spectra_SB3.py` generate synthetic spectra in `obs/` under the current working directory. Both use templates in `Models/`. For the installable binary workflow, run `cd example`, `python make_spectra_SB2.py`, then `saad quickstart.yaml`.

`example/disentangle_shift_and_add.py` runs the installable binary CLI with a YAML configuration. The original V2.0 Python-configured workflow remains in `example/legacy/` because it still supports triples. Edit `example/legacy/Input_disentangle.py`, then run `python example/legacy/disentangle_shift_and_add.py` from a directory containing `obs/`. It writes `Output/` in that directory. See [legacy documentation](example/legacy/README.md).
