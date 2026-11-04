"""Command line, validation, and optional processing checks."""

from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
import yaml

from saad.cli import main
from saad.binary import _clip_component
from saad.config import load_config, validate_config
from saad.processing import clean_cosmic_ray, normalise
from saad.runner import run


def test_cli_runs_yaml_configuration(case, capsys):
    path, _ = case
    _direct_file(path)
    main([str(path)])
    assert "K1=30.000 K2=50.000" in capsys.readouterr().out


def test_init_writes_bundled_example(tmp_path):
    target = tmp_path / "config.yaml"
    main(["--init", str(target)])
    assert target.read_text() == files("saad").joinpath("example.yaml").read_text()


def test_init_preserves_existing_file(tmp_path):
    target = tmp_path / "config.yaml"
    target.write_text("my config\n")
    with pytest.raises(FileExistsError):
        main(["--init", str(target)])


def test_bundled_example_matches_repository_copy():
    source = Path(__file__).parents[1] / "example" / "Input_disentangle.yaml"
    bundled = files("saad").joinpath("example.yaml")
    assert source.read_text() == bundled.read_text()
    assert load_config(bundled)["comp_num"] == 2


def _direct_file(path):
    data = yaml.safe_load(path.read_text())
    data.update(grid_dis=False, plot_itr=False, plot_extremes=False, plot_conv=False)
    path.write_text(yaml.safe_dump(data))


def test_cosmic_cleaner_removes_spike():
    spectrum = _spiked_spectrum()
    cleaned = clean_cosmic_ray(spectrum)
    assert cleaned[50, 1] == pytest.approx(1.0)


def _spiked_spectrum():
    wave = np.linspace(0, 10, 101)
    flux = np.ones_like(wave)
    flux[50] = 5
    return np.c_[wave, flux]


def test_normalisation_removes_linear_continuum():
    wave = np.linspace(0, 10, 101)
    spectrum = np.c_[wave, 1 + 0.1 * wave]
    result = normalise(spectrum, [0, 10])
    assert np.max(abs(result[:, 1] - 1)) < 0.05


def test_unknown_config_key_fails(case):
    path, _ = case
    with pytest.raises(ValueError, match="Unknown configuration"):
        validate_config(_unknown_config(path), path.parent)


def _unknown_config(path):
    config = load_config(path)
    config["grid_dist"] = True
    return config


def test_invalid_positive_window_fails(case):
    path, _ = case
    with pytest.raises(ValueError, match="Positive regions"):
        validate_config(_bad_positive_window(path), path.parent)


def _bad_positive_window(path):
    config = load_config(path)
    config["pos_lim_cond_A"] = [5000.0, 5001.0]
    return config


def test_positive_window_bypasses_absorption_clip():
    waves = np.array([5000.0, 5001.0])
    config = {"strict_neg_A": True, "pos_lim_cond_A": [[5000.9, 5001.1]], "force_neg_sigma": 2.0, "mean_s2n": 100.0}
    clipped = _clip_component(np.array([0.1, 0.1]), waves, config, "A")
    assert clipped == pytest.approx([0.02, 0.1])


def test_outside_wavelength_range_fails(case):
    path, _ = case
    with pytest.raises(ValueError, match="do not overlap"):
        run(_outside_config(path))


def _outside_config(path):
    config = load_config(path)
    config["ranges"] = [[6000.0, 6005.0]]
    return config
