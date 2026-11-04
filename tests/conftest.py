"""Small synthetic binary spectra shared by integration tests."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from saad.orbit import doppler_factor, velocities


ORBIT = {"Period": 10.0, "T0": 0.0, "ecc": 0.0, "omega": 0.0, "Gamma": 0.0, "K1": 30.0, "K2": 50.0}
CONFIG = {
    "obs_path": "obs", "output_path": "output", "lguess_vec": [0.35],
    "ranges": [[4998.0, 5002.0]], "grid_dis": True, "dense_k_arr": [3, 3],
    "ini_fac_K_arr": [2 / 3, 0.8], "fin_fac_K_arr": [4 / 3, 1.2],
    "itr_num_lim": 15, "num_itr_final": 40, "plot_extremes": True,
    "plot_conv": True, "plot_itr": True, "n_iteration_plot": 20,
}


def _line(wave, center, width, depth, velocity):
    rest = wave / doppler_factor(velocity)
    return -depth * np.exp(-0.5 * ((rest - center) / width) ** 2)


def _write_spectrum(folder, index, wave, velocity_a, velocity_b, nebular):
    flux = 1 + _line(wave, 5000.0, 0.25, 0.30, velocity_a)
    flux += _line(wave, 5000.55, 0.22, 0.16, velocity_b)
    flux += 0.07 * np.exp(-0.5 * ((wave - 5000.3) / 0.07) ** 2) if nebular else 0
    np.savetxt(folder / f"spec_{index}.txt", np.c_[wave, flux])


def _write_index(folder, dates):
    lines = ["MJD obsname"] + [f"{date} spec_{index}.txt" for index, date in enumerate(dates)]
    (folder / "ObsDat.txt").write_text("\n".join(lines) + "\n")


def _write_spectra(folder, nebular=False):
    wave, dates = np.arange(4995.0, 5005.0, 0.04), np.arange(0, 10, 1.25)
    velocity_a, velocity_b = velocities(dates, ORBIT)
    _write_epochs(folder, wave, velocity_a, velocity_b, nebular)
    _write_index(folder, dates)


def _write_epochs(folder, wave, velocity_a, velocity_b, nebular):
    for index, (a, b) in enumerate(zip(velocity_a, velocity_b)):
        _write_spectrum(folder, index, wave, a, b, nebular)


def _write_config(folder):
    config = {**CONFIG, "orbital_params": ORBIT}
    path = Path(folder) / "run.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture
def case(tmp_path):
    folder = tmp_path / "obs"
    folder.mkdir()
    _write_spectra(folder)
    return _write_config(tmp_path), ORBIT


@pytest.fixture
def nebular_case(case):
    path, orbit = case
    _write_spectra(path.parent / "obs", nebular=True)
    return path, orbit
