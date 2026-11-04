"""Keplerian radial velocities for the inner binary."""

import numpy as np


LIGHT_SPEED = 299792.458  # km/s


def eccentric_anomaly(mean_anomaly, eccentricity):
    estimate = np.asarray(mean_anomaly, dtype=float).copy()
    for _ in range(50):
        estimate -= (estimate - eccentricity * np.sin(estimate) - mean_anomaly) / (1 - eccentricity * np.cos(estimate))
    return estimate


def true_anomaly(times, orbit):
    mean = 2 * np.pi * (np.asarray(times) - orbit["T0"]) / orbit["Period"]
    eccentric = eccentric_anomaly(mean, orbit["ecc"])
    return 2 * np.arctan2(np.sqrt(1 + orbit["ecc"]) * np.sin(eccentric / 2),
                          np.sqrt(1 - orbit["ecc"]) * np.cos(eccentric / 2))


def velocities(times, orbit):
    angle = np.deg2rad(orbit["omega"])
    phase = np.cos(angle + true_anomaly(times, orbit)) + orbit["ecc"] * np.cos(angle)
    return orbit["Gamma"] + orbit["K1"] * phase, orbit["Gamma"] - orbit["K2"] * phase


def doppler_factor(velocity):
    """Convert radial velocity to relativistic wavelength factor."""
    beta = np.asarray(velocity) / LIGHT_SPEED
    return np.sqrt((1 + beta) / (1 - beta))
