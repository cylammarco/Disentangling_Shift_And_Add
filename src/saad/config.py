"""Load and validate configuration for the binary workflow."""

import math
from numbers import Real
from pathlib import Path

import yaml


REQUIRED = {"obs_path", "output_path", "orbital_params", "ranges", "lguess_vec"}
ORBIT_KEYS = {"Period", "T0", "ecc", "omega", "Gamma", "K1", "K2"}
DEFAULTS = {
    "star_name": "Target", "obs_format": "TXT", "obs_data_filename": "ObsDat.txt",
    "mjd_header_keyword": "MJD-OBS", "fig_type": "png", "comp_num": 2,
    "grid_dis": False, "dense_k_arr": [1, 1], "ini_fac_K_arr": [1.0, 1.0],
    "fin_fac_K_arr": [1.0, 1.0], "itr_num_lim": 50, "num_itr_final": 100,
    "inter_kind": "linear", "clean_cos": False, "renormalise": False,
    "norm_points": [], "neb_lines": False, "force_neg_sigma": 2.0,
    "strict_neg_A": False, "strict_neg_B": False, "pos_lim_cond_A": [],
    "pos_lim_cond_B": [], "range_str": "spectrum", "plot_extremes": False,
    "plot_fits": False, "plot_conv": False, "plot_itr": False,
    "n_iteration_plot": 50,
}


def load_config(path):
    source = Path(path).expanduser().resolve()
    data = yaml.safe_load(source.read_text())
    return validate_config(data, source.parent)


def validate_config(data, base_dir):
    _check_mapping(data)
    _check_values(data)
    config = {**DEFAULTS, **data}
    return _resolve_paths(config, base_dir)


def _check_mapping(data):
    if not isinstance(data, dict):
        raise ValueError("Configuration must contain a YAML mapping")
    _check_required(data)
    _check_unknown(data)


def _check_required(data):
    if missing := REQUIRED - data.keys():
        raise ValueError(f"Missing configuration keys: {', '.join(sorted(missing))}")


def _check_unknown(data):
    if unknown := data.keys() - (REQUIRED | DEFAULTS.keys()):
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")


def _check_values(data):
    _check_orbit(data["orbital_params"])
    _check_ranges(data["ranges"])
    _check_positive_ranges(data)
    _check_options(data)


def _check_orbit(orbit):
    if not isinstance(orbit, dict) or ORBIT_KEYS - orbit.keys():
        raise ValueError("orbital_params needs Period, T0, ecc, omega, Gamma, K1, K2")
    if not _valid_orbit_values(orbit):
        raise ValueError("Orbit values must be finite; Period, K1, K2 positive; ecc in [0, 1)")


def _valid_orbit_values(orbit):
    values = [orbit[key] for key in ORBIT_KEYS]
    if not all(isinstance(value, Real) and math.isfinite(value) for value in values):
        return False
    return orbit["Period"] > 0 and orbit["K1"] > 0 and orbit["K2"] > 0 and 0 <= orbit["ecc"] < 1


def _check_ranges(ranges):
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("ranges must be a nonempty list of wavelength pairs")
    if any(not _valid_range(pair) for pair in ranges):
        raise ValueError("Each range must contain two increasing wavelengths")


def _valid_range(pair):
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return False
    return all(isinstance(x, (int, float)) for x in pair) and pair[0] < pair[1]


def _check_positive_ranges(data):
    names = ("pos_lim_cond_A", "pos_lim_cond_B")
    if any(not _valid_optional_ranges(data.get(name, [])) for name in names):
        raise ValueError("Positive regions must be lists of wavelength pairs")


def _valid_optional_ranges(ranges):
    """Accept empty lists or increasing wavelength pairs."""
    return isinstance(ranges, list) and all(_valid_range(pair) for pair in ranges)


def _check_options(data):
    _check_components(data)
    _check_iterations(data)
    _check_grid(data)
    _check_interpolation(data)


def _check_components(data):
    if data.get("comp_num", 2) != 2:
        raise ValueError("Installable saad workflow currently supports two components")
    if not _valid_light_ratios(data["lguess_vec"]):
        raise ValueError("lguess_vec must contain one secondary light ratio in (0, 1)")


def _valid_light_ratios(ratios):
    if not isinstance(ratios, list) or len(ratios) != 1:
        return False
    return isinstance(ratios[0], Real) and 0 < ratios[0] < 1


def _check_iterations(data):
    if min(data.get("itr_num_lim", 1), data.get("num_itr_final", 1)) < 1:
        raise ValueError("Iteration counts must be positive")
    _check_processing(data)


def _check_processing(data):
    if data.get("renormalise") and not _valid_norm_points(data.get("norm_points", [])):
        raise ValueError("renormalise needs at least two increasing norm_points")
    if data.get("n_iteration_plot", 50) < 1:
        raise ValueError("n_iteration_plot must be positive")


def _valid_norm_points(points):
    if len(points) < 2:
        return False
    return all(a < b for a, b in zip(points, points[1:]))


def _check_grid(data):
    arrays = [data.get(key, DEFAULTS[key]) for key in ("dense_k_arr", "ini_fac_K_arr", "fin_fac_K_arr")]
    if not _valid_grid_arrays(arrays):
        raise ValueError("K1/K2 grid arrays must contain two valid values each")


def _valid_grid_arrays(arrays):
    if any(not isinstance(array, list) or len(array) != 2 for array in arrays):
        return False
    return all(isinstance(count, int) and count > 0 for count in arrays[0]) and _valid_grid_factors(arrays)


def _valid_grid_factors(arrays):
    """Require positive finite scale factors for both K axes."""
    return all(isinstance(value, Real) and math.isfinite(value) and value > 0
               for array in arrays[1:] for value in array)


def _check_interpolation(data):
    if data.get("inter_kind", "linear") not in {"linear", "cubic", "resample"}:
        raise ValueError("inter_kind must be linear, cubic, or resample")
    if data.get("obs_format", "TXT") not in {"TXT", "FITS"}:
        raise ValueError("obs_format must be TXT or FITS")


def _resolve_paths(config, base_dir):
    config["obs_path"] = str((base_dir / config["obs_path"]).resolve())
    config["output_path"] = str((base_dir / config["output_path"]).resolve())
    return config
