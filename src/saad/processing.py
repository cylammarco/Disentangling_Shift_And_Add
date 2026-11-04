"""Optional spectrum cleaning and continuum normalisation."""

import numpy as np
from scipy.ndimage import median_filter


def clean_cosmic_ray(spectrum, threshold=6):
    median = median_filter(spectrum[:, 1], size=7, mode="nearest")
    scatter = 1.4826 * np.median(abs(spectrum[:, 1] - median))
    cleaned = np.where(spectrum[:, 1] - median > threshold * max(scatter, 1e-6), median, spectrum[:, 1])
    return np.column_stack((spectrum[:, 0], cleaned))


def normalise(spectrum, points):
    anchors = np.asarray(points, dtype=float)
    levels = np.array([_local_level(spectrum, point) for point in anchors])
    continuum = np.interp(spectrum[:, 0], anchors, levels)
    return np.column_stack((spectrum[:, 0], spectrum[:, 1] / continuum))


def _local_level(spectrum, point):
    index = np.argmin(abs(spectrum[:, 0] - point))
    window = spectrum[max(0, index - 7) : index + 8, 1]
    return np.median(window)
