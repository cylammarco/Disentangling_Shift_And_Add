"""Input spectra and observation times."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import ascii, fits

from .file_handler import read_file
from .processing import clean_cosmic_ray, normalise


@dataclass(frozen=True)
class Observation:
    name: str
    time: float
    wave: np.ndarray
    flux: np.ndarray


def load_observations(config):
    entries = _entries(config)
    if not entries:
        raise ValueError(f"No spectra found in {config['obs_path']}")
    return [_observation(time, path, config) for time, path in entries]


def _entries(config):
    """Read observation dates from text index or FITS headers."""
    readers = {"TXT": _txt_entries, "FITS": _fits_entries}
    return readers[config["obs_format"]](config)


def _txt_entries(config):
    table = ascii.read(Path(config["obs_path"]) / config["obs_data_filename"])
    root = Path(config["obs_path"])
    return [(float(row["MJD"]), root / row["obsname"]) for row in table]


def _fits_entries(config):
    paths = sorted(Path(config["obs_path"]).glob("*.fits"))
    keyword = config["mjd_header_keyword"]
    return [(float(fits.getheader(path)[keyword]), path) for path in paths]


def _observation(time, path, config):
    spectrum = read_file(str(path))
    spectrum = spectrum[np.isfinite(spectrum).all(axis=1)]
    spectrum = _process(spectrum, config)
    return Observation(str(path), time, spectrum[:, 0], spectrum[:, 1])


def _process(spectrum, config):
    spectrum = clean_cosmic_ray(spectrum) if config["clean_cos"] else spectrum
    spectrum = normalise(spectrum, config["norm_points"]) if config["renormalise"] else spectrum
    return spectrum


def observation_weights(observations):
    variance = np.array([_noise(observation) ** 2 for observation in observations])
    inverse = 1 / np.maximum(variance, 1e-6)
    return inverse / inverse.sum()


def mean_s2n(observations):
    noises = np.array([_noise(observation) for observation in observations])
    return float(np.mean(1 / noises) * np.sqrt(len(noises)))


def _noise(observation):
    edges = np.r_[observation.flux[:20], observation.flux[-20:]]
    return max(float(np.std(edges)), 1e-3)
