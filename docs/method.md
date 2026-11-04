# How the binary workflow works

## Physical model

At epoch $j$, SAAD models a continuum-normalised observation as

$$
F_j(\lambda) \simeq 1 + A\!\left(\frac{\lambda}{D(v_{A,j})}\right)
                    + B\!\left(\frac{\lambda}{D(v_{B,j})}\right)
                    + N(\lambda).
$$

$A$ and $B$ are the two stellar contributions **relative to the combined continuum**, so their continuum level is zero. $N$ is an optional component fixed in the observer's wavelength frame, intended for stationary nebular emission. It is zero when `neb_lines: false`. The relativistic wavelength factor is $D(v)=\sqrt{(1+v/c)/(1-v/c)}$, with $c$ in km/s. The model assumes each stellar spectrum keeps the same shape in its own rest frame across observations.

Given period $P$, reference epoch $T_0$, eccentricity $e$, periastron angle $\omega$, and systemic velocity $\Gamma$, the code solves Kepler's equation and calculates

$$
v_A = \Gamma + K_1[\cos(\nu+\omega)+e\cos\omega],\qquad
v_B = \Gamma - K_2[\cos(\nu+\omega)+e\cos\omega].
$$

$\nu$ is true anomaly at the observation time. `omega` is supplied in degrees; $K_1$, $K_2$, and $\Gamma$ are in km/s. Observation times and $T_0$ must share the same time scale and units.

:::{figure} _static/figures/orbit.png
:alt: Opposing radial velocity curves for the example binary.
:width: 85%

Example radial velocity curves and observed epochs. Wider velocity separation gives more information for disentangling blended lines.
:::

## Preparation and iteration

1. Load the spectra and dates. Remove rows with non-finite wavelength or flux. Optionally clean positive spikes and renormalise the continuum.
2. Build a common wavelength grid over the intersection of the observed wavelength ranges, using the smallest input wavelength step.
3. Shift each observation to the primary and secondary rest frames. At each iteration, subtract the current secondary and stationary estimates, then co-add the residuals to update the primary. Repeat in the secondary frame. If requested, estimate the stationary component after subtracting both stars in the observer frame.
4. Weight each epoch by inverse estimated variance. The noise estimate is the standard deviation of the first and last 20 flux samples, with a floor of $10^{-3}$; weights are normalised to sum to one.
5. Stop after `num_itr_final` iterations. `result.history` stores the largest absolute change among components at each iteration. This is a convergence diagnostic, not an automatic stopping rule.

Out-of-range samples do not contribute to the rest-frame mean. Sampling between wavelengths uses `linear`, `cubic`, or flux-conserving `resample` interpolation, selected by `inter_kind`. The fit covers the **whole common wavelength grid**. `ranges` limits the wavelengths used to score candidate orbits, rather than limiting the saved spectra.

:::{figure} _static/figures/convergence.png
:alt: Maximum component change falling over 100 iterations in the example.
:width: 75%

Maximum component change across the 100 final iterations in the seeded example. Inspect the tail before choosing a smaller iteration count for new data.
:::

## Searching velocity amplitudes

With `grid_dis: true`, SAAD tries the Cartesian product of the configured $K_1$ and $K_2$ grids. Each point receives `itr_num_lim` quick iterations. The grid point with the smallest reduced chi-square score is selected; SAAD then repeats the fit there for `num_itr_final` iterations. With `grid_dis: false`, it uses the supplied $K_1$ and $K_2$ directly.

Each grid runs from `ini_fac_K_arr[i] * K_i` to `fin_fac_K_arr[i] * K_i`, inclusive, with `dense_k_arr[i]` points. A one-point grid uses the **initial** factor, so set that factor to `1.0` to hold an amplitude at its supplied value.

:::{figure} _static/figures/velocity_grid.png
:alt: Eleven by eleven reduced chi-square grid around the supplied velocity amplitudes.
:width: 75%

Reduced chi-square scores from a separate 11-by-11 scan of the seeded example observations. The white cross marks the selected point; the black plus marks the amplitudes used to generate the spectra. The quickstart uses a faster three-by-three grid for its component spectra. Inspect grid size and scoring assumptions before scientific use.
:::

The score uses residuals in `ranges`, dividing each observation by a noise estimate from its first 20 flux samples (floor $10^{-3}$). It divides the summed squared residual by the number of scored observed pixels minus two component spectra's scored pixels, with a minimum denominator of one. Consequently, the reported `reduced_chi2` is an operational fit score; interpret formal confidence intervals only after checking noise and model assumptions. The grid search does not estimate uncertainties for $K_1$ or $K_2$.

## Light ratios and interpretation

SAAD fits $A$ and $B$ as contributions to the observed total light. If the secondary supplies fraction $l_2$ of the continuum, the saved normalised spectra are

$$
F_A^{\mathrm{saved}}=1+\frac{A}{1-l_2},\qquad
F_B^{\mathrm{saved}}=1+\frac{B}{l_2}.
$$

`lguess_vec: [l_2]` sets this fraction; the binary spectra alone do not determine it. A wrong light ratio changes recovered line depths after scaling, even when the summed model fits the observations. Inspect the spectra and use external light-ratio information where possible.

## Assumptions and limits

- The installable workflow supports two stars. For triples, use the [archived V2.0 workflow](legacy.md).
- Stellar line profiles are assumed stable except for Doppler motion. Phase-dependent emission, eclipses, or changing continua can bias separation.
- Optional $N(\lambda)$ is stationary in the observer frame. Moving third-star lines cannot be represented by it.
- Continuum normalisation, wavelength calibration, orbit inputs, and wavelength coverage affect the result. Use the diagnostic plots and compare multiple spectral regions before scientific interpretation.
