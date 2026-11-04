"""Recover known stellar and nebular profiles from synthetic observations."""

import numpy as np
import pytest

from saad.config import load_config
from saad.runner import run


def _fit(path, mode="linear", nebular=False):
    config = load_config(path)
    config.update(grid_dis=False, plot_itr=False, plot_conv=False, plot_extremes=False,
                  inter_kind=mode, num_itr_final=100, neb_lines=nebular)
    return run(config)[0], path.parent / "output"


def _truth(waves):
    primary = -0.30 * np.exp(-0.5 * ((waves - 5000.0) / 0.25) ** 2)
    secondary = -0.16 * np.exp(-0.5 * ((waves - 5000.55) / 0.22) ** 2)
    nebula = 0.07 * np.exp(-0.5 * ((waves - 5000.3) / 0.07) ** 2)
    return primary, secondary, nebula


def _rms(actual, expected, waves, low=4998.0, high=5002.0):
    mask = (waves >= low) & (waves <= high)
    return np.sqrt(np.mean((actual[mask] - expected[mask]) ** 2))


def _profile_errors(result, truth):
    return [_rms(component, expected, result.waves) for component, expected in zip(
        (result.primary, result.secondary, result.nebula), truth)]


def _nebular_errors(result, saved, expected):
    recovered = _rms(result.nebula, expected, result.waves, 5000.1, 5000.5)
    exported = _rms(saved[:, 1], 1 + expected, saved[:, 0], 5000.1, 5000.5)
    return [recovered, exported]


def _all_nebular_errors(result, saved):
    truth = _truth(result.waves)
    return _profile_errors(result, truth)[:2] + _nebular_errors(result, saved, truth[2])


def _add_noise(path):
    rng = np.random.default_rng(192)
    for index, file in enumerate(sorted((path.parent / "obs").glob("spec_*.txt"))):
        spectrum = np.loadtxt(file)
        np.savetxt(file, np.c_[spectrum[:, 0], spectrum[:, 1] + rng.normal(0, 0.002 if index % 2 else 0.01, len(spectrum))])


@pytest.mark.parametrize("mode", ["linear", "cubic", "resample"])
def test_disentangled_profiles_match_truth(case, mode):
    result, _ = _fit(case[0], mode)
    truth = _truth(result.waves)
    assert max(_profile_errors(result, truth)[:2]) < 0.002


def test_seeded_noise_preserves_stellar_profiles(case):
    path, _ = case
    _add_noise(path)
    result, _ = _fit(path)
    assert max(_profile_errors(result, _truth(result.waves))[:2]) < 0.005


def test_nebular_profile_matches_truth(nebular_case):
    result, folder = _fit(nebular_case[0], nebular=True)
    saved = np.loadtxt(folder / "NebDIS.txt")
    assert np.array_equal(saved[:, 0], result.waves) and max(_all_nebular_errors(result, saved)) < 0.015


@pytest.mark.parametrize(("prefix", "fraction", "index"), [("ADIS", 0.65, 0), ("BDIS", 0.35, 1)])
def test_saved_stellar_spectrum_matches_truth(case, prefix, fraction, index):
    result, folder = _fit(case[0])
    saved = np.loadtxt(next(folder.glob(f"{prefix}*")))
    truth = 1 + _truth(result.waves)[index] / fraction
    assert np.array_equal(saved[:, 0], result.waves) and _rms(saved[:, 1], truth, saved[:, 0]) < 0.003
