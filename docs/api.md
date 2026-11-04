# Python API

The supported high-level entry points are `saad.load_config` and `saad.run`. Use them with a YAML file and inspect the returned result and selected orbit:

```python
from saad import load_config, run

config = load_config("example/quickstart.yaml")
result, orbit = run(config)
print(f"K1={orbit['K1']:.1f}, K2={orbit['K2']:.1f}, score={result.chi2:.3f}")
```

`run` has the same file-writing side effects as the CLI. With `grid_dis: false`, `orbit` is a copy of the supplied orbital mapping; with grid search enabled, it contains the selected `K1` and `K2`.

The command-line entry points are `saad --init PATH` to write a starter YAML file and `saad CONFIG.yaml` to run it. `python -m saad.cli CONFIG.yaml` is equivalent to the latter.

## Configuration and workflow

```{eval-rst}
.. autofunction:: saad.config.load_config
```

`load_config(path)` reads YAML, checks required and unknown keys, validates orbit and options, fills defaults, and resolves input/output paths relative to the YAML file. It returns a plain `dict`. Validation errors raise `ValueError`; filesystem and YAML parse errors propagate from their libraries.

```{eval-rst}
.. autofunction:: saad.config.validate_config
```

`validate_config(data, base_dir)` performs the same checks on an in-memory mapping and resolves paths against `base_dir`. Use it when constructing a configuration programmatically. Keep `base_dir` as a `pathlib.Path`.

```{eval-rst}
.. autofunction:: saad.runner.run
```

`run(config)` loads observations, optionally searches the velocity grid, fits the binary, saves spectra and selected diagnostics, and returns `(result, orbit)`. Pass a mapping returned by `load_config` or `validate_config`, so required defaults and paths are present.

### `FitResult`

`result` is a `saad.binary.FitResult` dataclass:

| Attribute | Meaning |
| --- | --- |
| `waves` | Common wavelength grid. |
| `primary`, `secondary` | Continuum-subtracted stellar contributions to total light, on `waves`; these are **not** the light-ratio-scaled saved spectra. |
| `nebula` | Continuum-subtracted stationary contribution on `waves`; all zeros if `neb_lines` is false. |
| `history` | List of largest component changes, one per final iteration. |
| `snapshots` | `(iteration, primary, secondary, nebula)` tuples captured when `plot_itr` is true. |
| `chi2` | Reduced fit score in configured `ranges`. |

To reconstruct the component spectra saved to disk, with `l2 = config["lguess_vec"][0]`, use `1 + result.primary / (1 - l2)` and `1 + result.secondary / l2`.

## Observation and orbit helpers

```{eval-rst}
.. autofunction:: saad.observations.load_observations
```

`load_observations(config)` returns a list of `Observation` dataclasses. Each has `name` (source path), `time`, `wave`, and `flux` arrays. It applies requested cosmic cleaning and renormalisation.

```{eval-rst}
.. autofunction:: saad.file_handler.read_file
```

`read_file(infile, col0=0, col1=1)` reads one spectrum without an observation index and returns an $N\times2$ wavelength/flux array. Pass `infile` as a **string filename**. The reader dispatches by extension or FITS instrument header. `col0` and `col1` select columns for fallback ASCII input.

```{eval-rst}
.. autofunction:: saad.orbit.velocities
```

`velocities(times, orbit)` returns primary and secondary radial-velocity arrays in km/s, using the orbit mapping described in the [configuration reference](configuration.md#required-keys).

```{eval-rst}
.. autofunction:: saad.orbit.doppler_factor
```

`doppler_factor(velocity)` returns the relativistic wavelength multiplier for velocities in km/s. It accepts a scalar or NumPy-compatible array.

## Spectrum helpers

```{eval-rst}
.. autofunction:: saad.binary.sample
```

`sample(waves, flux, target, mode, fill=0.0)` samples a component on requested wavelengths. Modes are `linear`, `cubic`, and `resample`. `fill` supplies values outside coverage. `resample` uses SpectRes and conserves flux under rebinning.

```{eval-rst}
.. autofunction:: saad.binary.observed_model
```

`observed_model(result, observation, factor_a, factor_b, mode)` shifts the fitted contributions into one observation's frame and returns the predicted continuum-normalised flux. Pass wavelength factors from `doppler_factor`, not velocities directly.

The older `saad.disentangle_functions.Disentangle` class remains importable for compatibility with earlier binary callers. It is separate from the YAML workflow, and the archived triple script uses its own module under `example/legacy/`. New binary integrations should use `load_config` and `run`.
