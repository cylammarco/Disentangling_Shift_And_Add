# Quickstart: synthetic SB2 spectra

This example uses the bundled `make_spectra_SB2.py` generator and TLUSTY model spectra in `Models/`. It produces 20 noisy observations of a binary, searches a small $K_1$/$K_2$ grid, and saves disentangled spectra. The generator writes into the **current working directory** and replaces matching files in `obs/`; use a fresh example directory when you have data there.

After [installing SAAD](installation.md), run from the repository checkout:

```bash
cd example
python make_spectra_SB2.py
saad quickstart.yaml
```

Running from `example/` matters: the generator creates `example/obs/`, and paths in `quickstart.yaml` resolve relative to the YAML file. The generator writes `obs/ObsDat.txt`, 20 two-column spectra, and two component templates. Its `ObsDat.txt` lists observation dates and spectrum filenames:

```text
MJD obsname
123.456 obs_0_V1_..._V2_...
234.567 obs_1_V1_..._V2_...
```

The actual dates and filenames vary because the generator adds random noise and draws random phases. The figures below use fixed random seeds (`23`) in `docs/generate_figures.py`. `quickstart.yaml` keeps the example's period, eccentricity, velocity amplitudes, secondary light fraction, and He I 4472 Å window. It uses three grid points per amplitude and 100 final iterations for a short run.

:::{figure} _static/figures/observed_fit.png
:alt: Two synthetic composite spectra at opposite velocity extremes, with fitted summed spectra.
:width: 95%

Observed composite spectra at two sampled phases, compared with the sum reconstructed from the two fitted components. The displayed wavelength window is the configured He I 4472 Å scoring range.
:::

## Inspect results

Files appear in `example/Output/`:

```text
ADIS_lguess2_K1K2=0.3_87.000_135.000.txt
BDIS_lguess2_K1K2=0.3_87.000_135.000.txt
HeI4472_grid_dis_K1K2.txt
fit_summary.json
DemoSB2_HeI4472_extremes.png
DemoSB2_convergence.png
```

This filename listing comes from the seeded documentation run. Your random realisation may choose another grid point and therefore another $K_1$/$K_2$ suffix. Each `ADIS` or `BDIS` text file contains wavelength and a normalised, light-ratio-scaled component spectrum. The output guide explains every file and the scaling.

:::{figure} _static/figures/components.png
:alt: Recovered primary and secondary spectra overlaid on the synthetic input templates.
:width: 95%

Disentangled spectra from the seeded example, compared with the templates used to create the observations. Small differences come from noise, interpolation, and finite iterations.
:::

## Use a Python script

From the repository root, after generating `example/obs/`:

```python
from saad import load_config, run

config = load_config("example/quickstart.yaml")
result, orbit = run(config)
print(orbit["K1"], orbit["K2"], result.chi2)
print(result.waves.shape, result.primary.shape, result.secondary.shape)
```

`run` also writes the files shown above. `result.primary` and `result.secondary` are contributions to the total normalised flux with continuum zero; the saved `ADIS` and `BDIS` spectra have continuum one and use the configured light fraction. See the [Python API](api.md) for returned fields and lower-level helpers.
