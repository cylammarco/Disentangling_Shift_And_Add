"""Configuration-driven binary disentangling workflow."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .binary import common_grid, fit_binary
from .diagnostics import render_plots
from .observations import load_observations, mean_s2n, observation_weights
from .orbit import velocities


@dataclass
class RunData:
    observations: list
    waves: np.ndarray
    weights: np.ndarray
    config: dict


def run(config):
    data = _prepare_run(config)
    orbit = search_orbit(data) if config["grid_dis"] else config["orbital_params"].copy()
    result = _final_fit(data, orbit)
    return _save_run(data, orbit, result)


def _prepare_run(config):
    observations = load_observations(config)
    _validate_observation_count(observations, config)
    waves = common_grid(observations)
    return _make_run_data(observations, waves, config)


def _validate_observation_count(observations, config):
    minimum = 3 if config["grid_dis"] else 2
    if len(observations) < minimum:
        raise ValueError(f"Need at least {minimum} observations for this workflow")


def _make_run_data(observations, waves, config):
    _validate_range_overlap(waves, config["ranges"])
    weights = observation_weights(observations)
    config = {**config, "mean_s2n": mean_s2n(observations)}
    return RunData(observations, waves, weights, config)


def _validate_range_overlap(waves, ranges):
    if not any(np.any((waves >= low) & (waves <= high)) for low, high in ranges):
        raise ValueError("Configured wavelength ranges do not overlap observations")


def search_orbit(data):
    grids = (_velocity_grid(data.config, "K1", 0), _velocity_grid(data.config, "K2", 1))
    scores = _grid_scores(data, *grids)
    return _best_orbit(data, grids, scores)


def _velocity_grid(config, key, index):
    value = config["orbital_params"][key]
    return np.linspace(config["ini_fac_K_arr"][index] * value,
                       config["fin_fac_K_arr"][index] * value, config["dense_k_arr"][index])


def _grid_scores(data, grid_a, grid_b):
    """Fit each velocity pair and store reduced chi-square values."""
    return np.array([[_score_pair(data, a, b) for b in grid_b] for a in grid_a])


def _score_pair(data, k1, k2):
    orbit = {**data.config["orbital_params"], "K1": k1, "K2": k2}
    velocity_a, velocity_b = velocities([obs.time for obs in data.observations], orbit)
    quick = {**data.config, "num_itr_final": data.config["itr_num_lim"]}
    return fit_binary(data.observations, data.waves, velocity_a, velocity_b, data.weights, quick).chi2


def _best_orbit(data, grids, scores):
    index = np.unravel_index(np.argmin(scores), scores.shape)
    _save_grid(data.config, grids, scores)
    orbit = {**data.config["orbital_params"], "K1": float(grids[0][index[0]]), "K2": float(grids[1][index[1]])}
    return orbit


def _save_grid(config, grids, scores):
    folder = Path(config["output_path"])
    folder.mkdir(parents=True, exist_ok=True)
    np.savetxt(folder / f"{config['range_str']}_grid_dis_K1K2.txt", scores,
               header=f"K1={grids[0].tolist()} K2={grids[1].tolist()}")


def _final_fit(data, orbit):
    velocity_a, velocity_b = velocities([obs.time for obs in data.observations], orbit)
    return fit_binary(data.observations, data.waves, velocity_a, velocity_b, data.weights, data.config)


def _save_run(data, orbit, result):
    folder = Path(data.config["output_path"])
    folder.mkdir(parents=True, exist_ok=True)
    _save_artifacts(folder, data, orbit, result)
    return result, orbit


def _save_artifacts(folder, data, orbit, result):
    _save_components(folder, data.config, orbit, result)
    _save_nebula(folder, data.config, result)
    _save_metrics(folder, orbit, result)
    render_plots(result, data.observations, orbit, data.config)


def _save_components(folder, config, orbit, result):
    secondary_ratio = config["lguess_vec"][0]
    suffix = f"{secondary_ratio}_{orbit['K1']:.3f}_{orbit['K2']:.3f}.txt"
    np.savetxt(folder / f"ADIS_lguess2_K1K2={suffix}", np.c_[result.waves, 1 + result.primary / (1 - secondary_ratio)])
    np.savetxt(folder / f"BDIS_lguess2_K1K2={suffix}", np.c_[result.waves, 1 + result.secondary / secondary_ratio])


def _save_metrics(folder, orbit, result):
    metrics = {"K1": orbit["K1"], "K2": orbit["K2"], "reduced_chi2": result.chi2}
    path = folder / "fit_summary.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n")


def _save_nebula(folder, config, result):
    if config["neb_lines"]:
        np.savetxt(folder / "NebDIS.txt", np.c_[result.waves, 1 + result.nebula])
