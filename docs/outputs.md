# Results and diagnostics

`run(config)` creates `output_path` and writes results there. Its filenames incorporate `star_name`, `range_str`, the secondary light fraction, or the selected velocity amplitudes as shown below. The examples here assume `example/quickstart.yaml`.

| File | Created when | Contents |
| --- | --- | --- |
| `ADIS_lguess2_K1K2=...txt` | Every run | Two columns: wavelength and primary normalised spectrum, scaled by $1-l_2$. |
| `BDIS_lguess2_K1K2=...txt` | Every run | Two columns: wavelength and secondary normalised spectrum, scaled by $l_2$. |
| `fit_summary.json` | Every run | Selected `K1`, `K2`, and `reduced_chi2`. |
| `<range_str>_grid_dis_K1K2.txt` | `grid_dis: true` | Reduced chi-square matrix; rows follow $K_1$ grid, columns follow $K_2$ grid. Header records both coordinate arrays. |
| `NebDIS.txt` | `neb_lines: true` | Wavelength and $1+N(\lambda)$ for stationary component. |
| `<star_name>_<range_str>_extremes.<fig_type>` | `plot_extremes: true` | Two observed/model plots at primary-velocity extremes. |
| `<star_name>_fit_000.<fig_type>`, etc. | `plot_fits: true` | One observed/model plot per epoch. |
| `<star_name>_convergence.<fig_type>` | `plot_conv: true` | Maximum component change by iteration. |
| `<star_name>_iteration_####.<fig_type>` | `plot_itr: true` | Component spectra at selected iterations. |

The output wavelength grid spans the common observed coverage. The configured `ranges` select pixels for scoring, not pixels written to the component spectra. The supplied `lguess_vec: [l_2]` determines the final primary and secondary scaling. See [light ratios](method.md#light-ratios-and-interpretation) before comparing line depths with stellar models.

## Read a component and summary

```python
import json
from pathlib import Path

import numpy as np

folder = Path("example/Output")
summary = json.loads((folder / "fit_summary.json").read_text())
primary_file = next(folder.glob("ADIS_lguess2_K1K2=*.txt"))
wave, primary_flux = np.loadtxt(primary_file, unpack=True)
print(summary["K1"], summary["K2"], wave.shape, primary_flux.min())
```

## Interpret the grid and convergence

:::{figure} _static/figures/velocity_grid.png
:alt: Reduced chi-square across an eleven by eleven grid in primary and secondary velocity amplitudes.
:width: 75%

An 11-by-11 scan of the seeded example. White cross marks its lowest score; black plus marks the amplitudes injected into the synthetic data. This is a separate refinement scan from the quickstart's three-by-three run, and it is not a measurement of amplitude uncertainty.
:::

:::{figure} _static/figures/convergence.png
:alt: Change in fitted spectra over iterations, decreasing toward zero.
:width: 75%

The change becomes small for this example, but SAAD always runs the configured number of final iterations. A low change does not by itself prove that the physical model or input orbit is correct.
:::

The `reduced_chi2` score compares reconstructed and observed flux within `ranges`, using a simple per-spectrum noise estimate. It is useful for ranking the configured grid and checking fit quality, but it is not a reported confidence interval.
