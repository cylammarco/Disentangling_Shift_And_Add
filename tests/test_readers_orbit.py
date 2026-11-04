"""Representative FITS wavelength and orbital-equation checks."""

import numpy as np
from astropy.io import fits

from saad.disentangle_functions import Disentangle
from saad.file_handler import read_HERMES
from saad.orbit import eccentric_anomaly


def test_kepler_solver_handles_eccentric_orbit():
    mean = np.linspace(0, 2 * np.pi, 12)
    original = mean.copy()
    solution = eccentric_anomaly(mean, 0.7)
    assert np.allclose(solution - 0.7 * np.sin(solution), mean) and np.array_equal(mean, original)


def test_legacy_disentangler_keeps_finite_pixels():
    spectrum = np.array([[1.0, 1.0], [2.0, np.nan], [3.0, 0.9]])
    solver = Disentangle([spectrum], {})
    assert solver.spec_list[0].tolist() == [[1.0, 1.0], [3.0, 0.9]]


def test_hermes_log_wavelength_axis(tmp_path):
    path = _hermes_file(tmp_path)
    wave, flux = read_HERMES(str(path))
    assert np.allclose(wave[:2], 5000 * np.exp([0, 0.001]))
    assert np.all(flux == 1)


def _hermes_file(folder):
    header = fits.Header({"INSTRUME": "HERMES", "CTYPE1": "log(wavelength)", "CRVAL1": np.log(5000), "CDELT1": 0.001})
    path = folder / "hermes.fits"
    fits.writeto(path, np.ones(5), header)
    return path
