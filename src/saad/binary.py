"""Weighted binary shift-and-add disentangling."""

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline
from spectresc import spectres

from .orbit import doppler_factor


@dataclass
class FitData:
    waves: np.ndarray
    observer: np.ndarray
    primary: np.ndarray
    secondary: np.ndarray
    factor_a: np.ndarray
    factor_b: np.ndarray
    weights: np.ndarray
    config: dict
    history: list = field(default_factory=list)
    snapshots: list = field(default_factory=list)


@dataclass
class FitResult:
    waves: np.ndarray
    primary: np.ndarray
    secondary: np.ndarray
    nebula: np.ndarray
    history: list
    snapshots: list
    chi2: float = 0.0


def sample(waves, flux, target, mode, fill=0.0):
    """Sample spectrum onto requested wavelengths."""
    methods = {"linear": _linear, "cubic": _cubic, "resample": _resample}
    return methods[mode](waves, flux, target, fill)


def _linear(waves, flux, target, fill):
    """Interpolate linearly without extrapolating."""
    return np.interp(target, waves, flux, left=fill, right=fill)


def _cubic(waves, flux, target, fill):
    values = CubicSpline(waves, flux, extrapolate=False)(target)
    return np.where(np.isfinite(values), values, fill)


def _resample(waves, flux, target, fill):
    """Resample flux with SpectRes C implementation."""
    return spectres(target, waves, flux, fill=fill, verbose=False)


def common_grid(observations):
    low, high = _common_bounds(observations)
    step = min(np.min(np.diff(obs.wave)) for obs in observations)
    grid = np.arange(low, high, step)
    return _validated_grid(grid)


def _common_bounds(observations):
    low = max(obs.wave[0] for obs in observations)
    high = min(obs.wave[-1] for obs in observations)
    return low, high


def _validated_grid(grid):
    if len(grid) < 3:
        raise ValueError("Observed spectra need at least three overlapping wavelengths")
    return grid


def fit_binary(observations, waves, velocity_a, velocity_b, weights, config):
    data = _prepare_fit(observations, waves, velocity_a, velocity_b, weights, config)
    primary, secondary, nebula = _fit_iterations(data, config["num_itr_final"])
    result = _make_result(data, primary, secondary, nebula, observations, velocity_a, velocity_b)
    return result


def _make_result(data, primary, secondary, nebula, observations, velocity_a, velocity_b):
    result = FitResult(data.waves, primary, secondary, nebula, data.history, data.snapshots)
    result.chi2 = reduced_chi2(result, observations, velocity_a, velocity_b,
                               data.config["inter_kind"], data.config["ranges"])
    return result


def _prepare_fit(observations, waves, velocity_a, velocity_b, weights, config):
    factor_a, factor_b = doppler_factor(velocity_a), doppler_factor(velocity_b)
    frames = _stellar_frames(observations, waves, factor_a, factor_b, config["inter_kind"])
    observer = _rest_stack(observations, waves, np.ones(len(observations)), config["inter_kind"])
    return FitData(waves, observer, *frames, factor_a, factor_b, weights, config)


def _stellar_frames(observations, waves, factor_a, factor_b, mode):
    primary = _rest_stack(observations, waves, factor_a, mode)
    secondary = _rest_stack(observations, waves, factor_b, mode)
    return primary, secondary


def _rest_stack(observations, waves, factors, mode):
    """Transform observations into one component's rest frame."""
    return np.stack([sample(obs.wave, obs.flux - 1, waves * factors[i], mode, np.nan)
                     for i, obs in enumerate(observations)])


def _fit_iterations(data, iterations):
    primary = secondary = nebula = np.zeros_like(data.waves)
    for _ in range(iterations):
        primary, secondary, nebula = _step(primary, secondary, nebula, data)
    return primary, secondary, nebula


def _step(primary, secondary, nebula, data):
    updated_a, updated_b = _estimate_stars(secondary, nebula, data)
    updated_nebula = _next_nebula(updated_a, updated_b, nebula, data)
    _record_step(data, primary, secondary, nebula, updated_a, updated_b, updated_nebula)
    return updated_a, updated_b, updated_nebula


def _next_nebula(primary, secondary, nebula, data):
    """Update stationary component only when requested."""
    return _estimate_nebula(primary, secondary, data) if data.config["neb_lines"] else nebula


def _estimate_stars(secondary, nebula, data):
    primary = _estimate_a(secondary, nebula, data)
    updated_secondary = _estimate_b(primary, nebula, data)
    return primary, updated_secondary


def _record_step(data, primary, secondary, nebula, updated_a, updated_b, updated_nebula):
    change = max(np.max(abs(updated_a - primary)), np.max(abs(updated_b - secondary)),
                 np.max(abs(updated_nebula - nebula)))
    data.history.append(float(change))
    _record_snapshot(data, updated_a, updated_b, updated_nebula)


def _record_snapshot(data, primary, secondary, nebula):
    iteration = len(data.history)
    step = data.config.get("n_iteration_plot", 50)
    if data.config.get("plot_itr") and (iteration % step == 0 or iteration == data.config["num_itr_final"]):
        data.snapshots.append((iteration, primary.copy(), secondary.copy(), nebula.copy()))


def _estimate_a(secondary, nebula, data):
    shifted = _component_stack(secondary, data.waves, data.factor_a / data.factor_b, data.config["inter_kind"])
    contamination = _component_stack(nebula, data.waves, data.factor_a, data.config["inter_kind"])
    updated = _weighted_mean(data.primary - shifted - contamination, data.weights)
    return _clip_component(updated, data.waves, data.config, "A")


def _estimate_b(primary, nebula, data):
    shifted = _component_stack(primary, data.waves, data.factor_b / data.factor_a, data.config["inter_kind"])
    contamination = _component_stack(nebula, data.waves, data.factor_b, data.config["inter_kind"])
    updated = _weighted_mean(data.secondary - shifted - contamination, data.weights)
    return _clip_component(updated, data.waves, data.config, "B")


def _estimate_nebula(primary, secondary, data):
    shifted_a = _component_stack(primary, data.waves, 1 / data.factor_a, data.config["inter_kind"])
    shifted_b = _component_stack(secondary, data.waves, 1 / data.factor_b, data.config["inter_kind"])
    return _weighted_mean(data.observer - shifted_a - shifted_b, data.weights)


def _component_stack(component, waves, ratios, mode):
    """Move component between stellar rest frames."""
    return np.stack([sample(waves, component, waves * ratio, mode) for ratio in ratios])


def _weighted_mean(values, weights):
    finite = np.isfinite(values)
    numerator = np.sum(np.where(finite, values, 0) * weights[:, None], axis=0)
    denominator = np.sum(finite * weights[:, None], axis=0)
    return numerator / np.maximum(denominator, 1e-12)


def _clip_component(values, waves, config, label):
    if not config.get(f"strict_neg_{label}"):
        return values
    emission = _emission_mask(waves, config.get(f"pos_lim_cond_{label}", []))
    return np.where(emission, values, np.minimum(values, _positive_limit(config)))


def _positive_limit(config):
    """Upper continuum offset allowed for absorption components."""
    return config["force_neg_sigma"] / config.get("mean_s2n", 100)


def _emission_mask(waves, ranges):
    mask = np.zeros(len(waves), dtype=bool)
    for low, high in ranges:
        mask |= (waves >= low) & (waves <= high)
    return mask


def observed_model(result, observation, factor_a, factor_b, mode):
    primary = sample(result.waves, result.primary, observation.wave / factor_a, mode)
    secondary = sample(result.waves, result.secondary, observation.wave / factor_b, mode)
    nebula = sample(result.waves, result.nebula, observation.wave, mode)
    return 1 + primary + secondary + nebula


def reduced_chi2(result, observations, velocity_a, velocity_b, mode, ranges=None):
    factors = zip(doppler_factor(velocity_a), doppler_factor(velocity_b))
    chi2 = sum(_chi2_one(result, obs, a, b, mode, ranges) for obs, (a, b) in zip(observations, factors))
    return float(chi2 / _degrees_of_freedom(result, observations, ranges))


def _degrees_of_freedom(result, observations, ranges):
    observed = sum(_range_mask(obs.wave, ranges).sum() for obs in observations)
    components = 2 * _range_mask(result.waves, ranges).sum()
    return max(1, observed - components)


def _chi2_one(result, observation, factor_a, factor_b, mode, ranges):
    mask = _range_mask(observation.wave, ranges)
    model = observed_model(result, observation, factor_a, factor_b, mode)
    noise = max(np.std(observation.flux[: min(20, len(observation.flux))]), 1e-3)
    return np.sum(((observation.flux[mask] - model[mask]) / noise) ** 2)


def _range_mask(waves, ranges):
    if ranges is None:
        return np.ones(len(waves), dtype=bool)
    return _emission_mask(waves, ranges)
