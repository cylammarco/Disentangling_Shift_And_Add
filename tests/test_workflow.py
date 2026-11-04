"""Configuration and synthetic-spectrum integration tests."""

import ast
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from saad.binary import sample
from saad.config import load_config, validate_config
from saad.observations import _observation, load_observations
from saad.runner import run


NEW_MODULES = ("binary.py", "cli.py", "config.py", "diagnostics.py", "observations.py",
               "orbit.py", "processing.py", "runner.py")


@pytest.fixture
def fitted_case(case):
    path, _ = case
    result, orbit = run(load_config(path))
    return path.parent, result, orbit


def test_grid_recovers_injected_velocities(fitted_case):
    _, result, orbit = fitted_case
    assert orbit["K1"] == pytest.approx(30.0)
    assert orbit["K2"] == pytest.approx(50.0)
    assert result.chi2 < 2


def test_iteration_snapshots_are_sampled(fitted_case):
    _, result, _ = fitted_case
    assert len(result.snapshots) == 2
    assert all(isinstance(change, float) for change in result.history)


@pytest.mark.parametrize("mode", ["linear", "cubic", "resample"])
def test_direct_fit_uses_supplied_orbit(case, mode):
    path, expected = case
    config = _direct_config(path, mode)
    result, orbit = run(config)
    assert orbit == expected and np.isfinite(result.chi2) and not result.snapshots


def _direct_config(path, mode):
    config = load_config(path)
    config.update(grid_dis=False, plot_itr=False, inter_kind=mode)
    return config


def test_run_writes_spectra_grid_and_plots(fitted_case):
    root, result, _ = fitted_case
    names = {path.name for path in (root / "output").iterdir()}
    assert any(name.startswith("ADIS_") for name in names)
    _assert_artifacts(names, result)


def _assert_artifacts(names, result):
    required = {"fit_summary.json", "spectrum_grid_dis_K1K2.txt",
                "Target_convergence.png", "Target_spectrum_extremes.png"}
    assert required <= names
    assert len(result.history) == 40


def test_nebular_component_is_recovered(nebular_case):
    path, _ = nebular_case
    result, config = _run_nebular(path)
    assert 0.04 < result.nebula.max() < 0.12
    assert (Path(config["output_path"]) / "NebDIS.txt").exists()


def _run_nebular(path):
    config = load_config(path)
    config["neb_lines"] = True
    result, _ = run(config)
    return result, config


@pytest.mark.parametrize("mode", ["linear", "cubic", "resample"])
def test_sampling_modes_preserve_constant_spectrum(mode):
    waves = np.arange(5000.0, 5005.0, 0.1)
    sampled = sample(waves, np.ones_like(waves), waves[2:-2], mode)
    assert np.allclose(sampled, 1)


def test_config_paths_are_relative_to_yaml(case):
    path, _ = case
    config = load_config(path)
    assert Path(config["obs_path"]) == path.parent / "obs"
    assert Path(config["output_path"]) == path.parent / "output"


def test_config_rejects_invalid_range(case):
    path, _ = case
    with pytest.raises(ValueError, match="increasing"):
        validate_config(_invalid_range(path), path.parent)


def _invalid_range(path):
    data = load_config(path)
    data["ranges"] = [[5002.0, 4998.0]]
    return data


def test_fits_observation_uses_header_date(tmp_path, case):
    path, _ = case
    config = _fits_config(tmp_path, path)
    observation = load_observations(config)[0]
    assert observation.time == 2.5 and observation.wave[0] == 5000.0


def test_nonfinite_spectrum_rows_are_removed(tmp_path):
    path = tmp_path / "spectrum.txt"
    np.savetxt(path, [[1, 1], [2, np.nan], [3, np.inf], [4, 1]])
    observation = _observation(0, path, {"clean_cos": False, "renormalise": False})
    assert observation.wave.tolist() == [1, 4]


def _fits_config(tmp_path, path):
    header = fits.Header({"CRVAL1": 5000.0, "CDELT1": 0.1, "MJD-OBS": 2.5})
    fits.writeto(tmp_path / "sample.fits", np.ones(20), header)
    return {**load_config(path), "obs_format": "FITS", "obs_path": str(tmp_path)}


def test_new_functions_have_three_to_five_lines():
    root = Path(__file__).parents[1] / "src" / "saad"
    paths = [root / name for name in NEW_MODULES] + list(Path(__file__).parent.glob("*.py"))
    functions = _functions(paths)
    assert all(3 <= node.end_lineno - node.lineno + 1 <= 5 for node in functions)


def test_new_generic_reader_has_three_to_five_lines():
    source = Path(__file__).parents[1] / "src" / "saad" / "file_handler.py"
    created = (node for node in _functions([source]) if node.name in {"read_generic_fits", "_linear_wave"})
    assert all(3 <= node.end_lineno - node.lineno + 1 <= 5 for node in created)


def _functions(paths):
    """Yield every function in package and tests."""
    return (node for path in paths for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.FunctionDef))
