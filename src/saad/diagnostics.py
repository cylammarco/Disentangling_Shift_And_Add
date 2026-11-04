"""Optional plots for fitted binary spectra."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .binary import sample
from .orbit import doppler_factor, velocities


def render_plots(result, observations, orbit, config):
    velocity_a, velocity_b = velocities([obs.time for obs in observations], orbit)
    _plot_selected(result, observations, velocity_a, velocity_b, config)


def _plot_selected(result, observations, velocity_a, velocity_b, config):
    for option, plot in PLOTS.items():
        if config[option]:
            plot(result, observations, velocity_a, velocity_b, config)


def plot_extremes(result, observations, velocity_a, velocity_b, config):
    indices = [int(np.argmin(velocity_a)), int(np.argmax(velocity_a))]
    figure, axes = plt.subplots(2, 1, figsize=(8, 8))
    _draw_extremes(axes, indices, result, observations, velocity_a, velocity_b, config)
    _save_figure(figure, config, f"{config['star_name']}_{config['range_str']}_extremes")


def _draw_extremes(axes, indices, result, observations, velocity_a, velocity_b, config):
    for axis, index in zip(axes, indices):
        _plot_epoch(axis, result, observations[index], velocity_a[index], velocity_b[index], config["inter_kind"])


def plot_fits(result, observations, velocity_a, velocity_b, config):
    for index, (observation, a, b) in enumerate(zip(observations, velocity_a, velocity_b)):
        _plot_single_epoch(result, observation, a, b, config, index)


def _plot_single_epoch(result, observation, velocity_a, velocity_b, config, index):
    figure, axis = plt.subplots()
    _plot_epoch(axis, result, observation, velocity_a, velocity_b, config["inter_kind"])
    _save_figure(figure, config, f"{config['star_name']}_fit_{index:03d}")


def _plot_epoch(axis, result, observation, velocity_a, velocity_b, mode):
    for label, curve in _curves(result, observation, velocity_a, velocity_b, mode).items():
        axis.plot(observation.wave, curve, label=label)
    axis.set(title=Path(observation.name).name, xlabel="Wavelength", ylabel="Normalised flux")
    axis.legend()


def _curves(result, observation, velocity_a, velocity_b, mode):
    primary = sample(result.waves, result.primary, observation.wave / doppler_factor(velocity_a), mode)
    secondary = sample(result.waves, result.secondary, observation.wave / doppler_factor(velocity_b), mode)
    nebula = sample(result.waves, result.nebula, observation.wave, mode)
    return _curve_map(observation.flux, primary, secondary, nebula)


def _curve_map(observed, primary, secondary, nebula):
    """Group observed and model curves for one epoch."""
    return {"observation": observed, "sum": 1 + primary + secondary + nebula,
            "primary": 1 + primary, "secondary": 1 + secondary, "nebula": 1 + nebula}


def plot_convergence(result, observations, velocity_a, velocity_b, config):
    figure, axis = plt.subplots()
    axis.plot(result.history)
    axis.set(xlabel="Iteration", ylabel="Maximum change")
    _save_figure(figure, config, f"{config['star_name']}_convergence")


def plot_iterations(result, observations, velocity_a, velocity_b, config):
    for index, primary, secondary, nebula in result.snapshots:
        _plot_snapshot(result.waves, primary, secondary, nebula, index, config)


def _plot_snapshot(waves, primary, secondary, nebula, index, config):
    figure, axis = plt.subplots()
    _snapshot_curves(axis, waves, primary, secondary, nebula)
    _save_figure(figure, config, f"{config['star_name']}_iteration_{index:04d}")


def _snapshot_curves(axis, waves, primary, secondary, nebula):
    axis.plot(waves, 1 + primary, label="primary")
    axis.plot(waves, 1 + secondary, label="secondary")
    axis.plot(waves, 1 + nebula, label="nebula")
    axis.legend()


def _save_figure(figure, config, name):
    figure.tight_layout()
    figure.savefig(Path(config["output_path"]) / f"{name}.{config['fig_type']}")
    plt.close(figure)


PLOTS = {"plot_extremes": plot_extremes, "plot_fits": plot_fits,
         "plot_conv": plot_convergence, "plot_itr": plot_iterations}
