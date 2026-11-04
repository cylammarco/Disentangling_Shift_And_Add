# Data and configuration reference

The CLI accepts one YAML mapping. Generate the fully commented example with `saad --init config.yaml`, or use `example/quickstart.yaml` for a short run. Unknown keys and missing required keys raise `ValueError`. **Relative `obs_path` and `output_path` are resolved against the YAML file's directory**, regardless of the shell's working directory.

## Input spectra

### Text files

Set `obs_format: TXT` (the default). Place `ObsDat.txt` inside `obs_path` unless `obs_data_filename` names another index. Its first row must supply `MJD` and `obsname` columns:

```text
MJD obsname
59000.0 spectrum_01.txt
59001.5 spectrum_02.txt
```

Each `obsname` is resolved relative to `obs_path`; absolute paths also work. Each spectrum must provide wavelength and normalised flux columns in increasing wavelength order. SAAD's file reader accepts common ASCII extensions, including `.txt`, `.dat`, and `.ascii`. Time values need only match the units and reference epoch used by `orbital_params`; the column is still named `MJD`.

### FITS files

Set `obs_format: FITS`. SAAD loads `*.fits` files from `obs_path`, taking each observation time from the header key named by `mjd_header_keyword` (default `MJD-OBS`). The generic reader supports a one-dimensional flux array with linear wavelength coordinates from `CRVAL1`, `CDELT1` or `CD1_1`, and optional `CRPIX1`. Specialised readers cover MUSE, HERMES, FEROS, XSHOOTER, UVES, GIRAFFE, ESPCOUDE, COS, and STIS headers. Verify that your FITS headers and wavelength units match `ranges`.

At least two observations are required for a fixed orbit and at least three for grid search. Spectra need at least three overlapping wavelength samples. Rows containing NaN or infinity are discarded before fitting.

## Required keys

| Key | Type | Meaning |
| --- | --- | --- |
| `obs_path` | path | Directory containing spectra and, for text input, `ObsDat.txt`. |
| `output_path` | path | Directory where result files and selected plots are written. Created if needed. |
| `orbital_params` | mapping | Keys `Period`, `T0`, `ecc`, `omega`, `Gamma`, `K1`, `K2`. See below. |
| `ranges` | list of `[low, high]` pairs | Wavelength intervals used for orbit scoring and reported reduced chi-square. Each pair must increase. |
| `lguess_vec` | one-element list | Secondary continuum fraction, strictly between zero and one. Example: `[0.3]`. |

`orbital_params` uses `Period` and `T0` in the same units as observation dates, `ecc` in $[0,1)$, `omega` in degrees, and `Gamma`, `K1`, `K2` in km/s. `Period`, `K1`, and `K2` must be positive. The CLI can search only `K1` and `K2`; other entries remain fixed.

## Optional keys

### Files, naming, and components

| Key | Default | Meaning |
| --- | --- | --- |
| `star_name` | `Target` | Prefix for plot filenames. |
| `obs_format` | `TXT` | `TXT` or `FITS`. |
| `obs_data_filename` | `ObsDat.txt` | Text observation index filename. |
| `mjd_header_keyword` | `MJD-OBS` | FITS header key for observation time. |
| `range_str` | `spectrum` | Label used in grid and some plot filenames. |
| `comp_num` | `2` | Must remain `2` in the installable workflow. |
| `neb_lines` | `false` | Fit a stationary spectral component in addition to the two stars. |

### Orbit search and iterations

| Key | Default | Meaning |
| --- | --- | --- |
| `grid_dis` | `false` | Search $K_1$/$K_2$ grid when true; otherwise use supplied amplitudes. |
| `dense_k_arr` | `[1, 1]` | Number of grid points for $K_1$ and $K_2$. |
| `ini_fac_K_arr` | `[1.0, 1.0]` | Starting factors multiplied by supplied $K_1$ and $K_2$. |
| `fin_fac_K_arr` | `[1.0, 1.0]` | Ending factors multiplied by supplied amplitudes. |
| `itr_num_lim` | `50` | Iterations used at each trial grid point. |
| `num_itr_final` | `100` | Iterations at the selected or supplied orbit. |
| `inter_kind` | `linear` | `linear`, `cubic`, or flux-conserving `resample`. |

Grid arrays must each have two values. Grid counts and iteration counts must be positive. The grid endpoints are included; `dense_k_arr: [1, 1]` samples only the starting factors. A wider or finer grid costs more because each pair requires another fit. See [method](method.md) for the scoring procedure.

### Processing, positivity, and plots

| Key | Default | Meaning |
| --- | --- | --- |
| `clean_cos` | `false` | Replace large positive flux spikes with a local median before fitting. |
| `renormalise` | `false` | Divide by a piecewise-linear continuum estimated at `norm_points`. |
| `norm_points` | `[]` | Increasing wavelength anchors; at least two required when `renormalise` is true. |
| `strict_neg_A`, `strict_neg_B` | `false` | Restrict positive stellar contribution outside allowed emission windows. |
| `force_neg_sigma` | `2.0` | Positive allowance, in units of the estimated mean signal-to-noise denominator. |
| `pos_lim_cond_A`, `pos_lim_cond_B` | `[]` | Wavelength intervals where the corresponding stellar contribution may be positive. |
| `fig_type` | `png` | Matplotlib output extension for selected plots. |
| `plot_extremes` | `false` | Plot observations and model at the two primary-velocity extremes. |
| `plot_fits` | `false` | Write one observed-versus-model plot per epoch. |
| `plot_conv` | `false` | Plot largest component change by iteration. |
| `plot_itr` | `false` | Save component snapshots at configured iteration intervals. |
| `n_iteration_plot` | `50` | Interval for snapshots when `plot_itr` is enabled. |

With `strict_neg_A` or `strict_neg_B`, positive continuum-subtracted values outside `pos_lim_cond_A` or `pos_lim_cond_B` are clipped to `force_neg_sigma / mean_s2n`. This can limit positive wings in absorption-dominated spectra; include known stellar emission lines in the positive windows or turn off clipping for those components. The [quickstart](quickstart.md) enables clipping for its absorption-line templates.
