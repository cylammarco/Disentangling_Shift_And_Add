"""Regenerate documentation figures from the shipped SB2 example."""

import os
import random
import runpy
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from saad.binary import observed_model
from saad.config import load_config
from saad.observations import load_observations
from saad.orbit import doppler_factor, velocities
from saad.runner import run


ROOT = Path(__file__).resolve().parents[1]
FIGURES = Path(__file__).resolve().parent / "_static" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE = "#1764a3", "#c35320"


def _save(figure, name):
    figure.tight_layout()
    figure.savefig(FIGURES / name, dpi=180)
    plt.close(figure)


def _grid_values(config, key, index):
    base = config["orbital_params"][key]
    low = config["ini_fac_K_arr"][index] * base
    high = config["fin_fac_K_arr"][index] * base
    return np.linspace(low, high, config["dense_k_arr"][index])


with TemporaryDirectory() as temporary:
    folder = Path(temporary)
    previous = Path.cwd()
    try:
        os.chdir(folder)
        random.seed(23)
        np.random.seed(23)
        runpy.run_path(str(ROOT / "example" / "make_spectra_SB2.py"), run_name="__main__")
    finally:
        os.chdir(previous)

    config = load_config(ROOT / "example" / "quickstart.yaml")
    config.update(obs_path=str(folder / "obs"), output_path=str(folder / "Output"))
    result, orbit = run(config)
    observations = load_observations(config)
    observed_times = np.array([observation.time for observation in observations])
    velocity_a, velocity_b = velocities(observed_times, orbit)

    phase = np.linspace(0, 1, 500)
    orbit_a, orbit_b = velocities(orbit["T0"] + phase * orbit["Period"], orbit)
    observed_phase = ((observed_times - orbit["T0"]) / orbit["Period"]) % 1
    figure, axis = plt.subplots(figsize=(7.5, 4.3))
    axis.plot(phase, orbit_a, color=BLUE, label="Primary orbit")
    axis.plot(phase, orbit_b, color=ORANGE, label="Secondary orbit")
    axis.scatter(observed_phase, velocity_a, color=BLUE, edgecolor="white", zorder=3)
    axis.scatter(observed_phase, velocity_b, color=ORANGE, edgecolor="white", zorder=3)
    axis.set(xlabel="Orbital phase", ylabel="Radial velocity (km/s)", xlim=(0, 1))
    axis.legend(loc="upper right")
    axis.grid(alpha=0.2)
    _save(figure, "orbit.png")

    indices = [int(np.argmin(velocity_a)), int(np.argmax(velocity_a))]
    figure, axes = plt.subplots(2, 1, figsize=(7.5, 5.7), sharex=True, sharey=True)
    for axis, index in zip(axes, indices):
        observation = observations[index]
        mask = (observation.wave >= 4457) & (observation.wave <= 4489)
        model = observed_model(result, observation, doppler_factor(velocity_a[index]),
                               doppler_factor(velocity_b[index]), config["inter_kind"])
        axis.plot(observation.wave[mask], observation.flux[mask], color="0.35", lw=1, label="Observed")
        axis.plot(observation.wave[mask], model[mask], color=ORANGE, lw=1.5, label="Fitted sum")
        axis.set(ylabel="Normalised flux", title=f"Phase {observed_phase[index]:.2f}; primary RV {velocity_a[index]:+.0f} km/s")
        axis.legend(loc="lower right")
    axes[-1].set_xlabel("Wavelength (Å)")
    _save(figure, "observed_fit.png")

    grid_config = {**config, "dense_k_arr": [11, 11], "output_path": str(folder / "GridOutput"),
                   "num_itr_final": 8, "plot_extremes": False, "plot_conv": False}
    _, grid_orbit = run(grid_config)
    scores = np.loadtxt(folder / "GridOutput" / "HeI4472_grid_dis_K1K2.txt")
    assert scores.shape == (11, 11)
    k1 = _grid_values(grid_config, "K1", 0)
    k2 = _grid_values(grid_config, "K2", 1)
    k1_half = (k1[1] - k1[0]) / 2
    k2_half = (k2[1] - k2[0]) / 2
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    image = axis.imshow(scores.T, origin="lower", extent=(k1[0] - k1_half, k1[-1] + k1_half,
                                                        k2[0] - k2_half, k2[-1] + k2_half),
                        aspect="auto", interpolation="nearest", cmap="viridis")
    axis.scatter([grid_orbit["K1"]], [grid_orbit["K2"]], marker="x", s=95, color="white", lw=2, label="Best grid point")
    axis.scatter([config["orbital_params"]["K1"]], [config["orbital_params"]["K2"]],
                 marker="+", s=100, color="black", lw=2, label="Input amplitudes")
    axis.set(xlabel="Primary K1 (km/s)", ylabel="Secondary K2 (km/s)", title="11 × 11 velocity grid")
    axis.legend(loc="upper right")
    figure.colorbar(image, ax=axis, label="Reduced χ²")
    _save(figure, "velocity_grid.png")

    templates = [np.loadtxt(folder / "obs" / name) for name in ("Atemp.txt", "Btemp.txt")]
    saved = [np.loadtxt(next((folder / "Output").glob(f"{prefix}*"))) for prefix in ("ADIS", "BDIS")]
    figure, axes = plt.subplots(2, 1, figsize=(7.5, 5.7), sharex=True)
    for axis, template, recovered, title, color in zip(axes, templates, saved, ("Primary", "Secondary"), (BLUE, ORANGE)):
        truth_mask = (template[:, 0] >= 4457) & (template[:, 0] <= 4489)
        fit_mask = (recovered[:, 0] >= 4457) & (recovered[:, 0] <= 4489)
        axis.plot(template[truth_mask, 0], template[truth_mask, 1], color="0.4", lw=1.4, label="Input template")
        axis.plot(recovered[fit_mask, 0], recovered[fit_mask, 1], color=color, lw=1.3, label="Disentangled")
        axis.set(ylabel="Normalised flux", title=title)
        axis.legend(loc="lower right")
    axes[-1].set_xlabel("Wavelength (Å)")
    _save(figure, "components.png")

    shutil.copyfile(folder / "Output" / "DemoSB2_convergence.png", FIGURES / "convergence.png")
