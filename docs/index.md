# SAAD: shift-and-add spectral disentangling

SAAD separates the spectra of two stars in a spectroscopic binary from observations taken at different orbital phases. The installable Python package reads normalised spectra, computes orbital radial velocities, iteratively estimates each star's spectrum, and optionally searches a grid of velocity amplitudes $K_1$ and $K_2$. It can also estimate emission that remains stationary in the observer's wavelength frame.

The result is a pair of component spectra, diagnostic plots, and a fit summary. You supply the orbital period, epoch, eccentricity, argument of periastron, systemic velocity, and a light ratio. The grid search can refine $K_1$ and $K_2$; it does not fit the other orbital parameters.

:::{figure} _static/figures/orbit.png
:alt: Primary and secondary radial velocity curves with sampled observation phases.
:width: 90%

Radial velocities for the bundled synthetic SB2 example. Dots mark the 20 observed epochs. Opposite Doppler shifts make it possible to separate the two spectra.
:::

## Choose a workflow

- **Installable `saad` CLI and Python API:** binary spectra, optional stationary nebular emission, and $K_1$/$K_2$ grid search. Start with the [quickstart](quickstart.md).
- **Original V2.0 Python scripts:** legacy binary or triple analysis. Read the [legacy workflow](legacy.md).

The installable workflow currently supports **two stellar components**. Triple disentangling remains in the archived V2.0 script. These workflows use different configuration formats and output conventions.

## What these pages cover

- [Installation](installation.md) and a runnable [SB2 example](quickstart.md).
- [Method and assumptions](method.md), including shifting, weighting, fitting, and light ratios.
- Complete [input and configuration reference](configuration.md) and [output guide](outputs.md).
- [Python API](api.md) for integrating SAAD into scripts.

The implementation follows iterative shift-and-add disentangling as discussed by González & Levato (2006, *A&A*, 448, 283) and in Shenar et al. (2020, *A&A*, 639, A6; 2022, *A&A*, 665, A148). Consult those papers when using derived spectra or orbital amplitudes in scientific work.

```{toctree}
:maxdepth: 2
:hidden:

installation
quickstart
method
configuration
outputs
api
legacy
```
