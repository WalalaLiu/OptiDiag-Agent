"""Diffraction-specific metrics built on basic image features."""

from __future__ import annotations

from typing import Dict

import numpy as np

from optidiag.analysis.metrics import compute_basic_metrics
from optidiag.utils.image_io import to_float_gray


def fringe_visibility(image: np.ndarray) -> float:
    """Estimate Michelson-like visibility from the central intensity profile."""
    arr = to_float_gray(image)
    h = arr.shape[0]
    band = arr[max(0, h // 2 - 2) : min(h, h // 2 + 3), :]
    profile = np.mean(band, axis=0)
    p5, p995 = np.percentile(profile, [5, 99.5])
    return float(np.clip((p995 - p5) / (p995 + p5 + 1e-6), 0.0, 1.0))


def orientation_estimate_deg(image: np.ndarray) -> float:
    """Estimate dominant bright-pattern orientation from second moments."""
    arr = to_float_gray(image)
    y_idx, x_idx = np.indices(arr.shape)
    weights = arr - np.percentile(arr, 50)
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total <= 1e-8:
        return 0.0
    x_mean = float((x_idx * weights).sum() / total)
    y_mean = float((y_idx * weights).sum() / total)
    x = x_idx - x_mean
    y = y_idx - y_mean
    mu20 = float((weights * x * x).sum() / total)
    mu02 = float((weights * y * y).sum() / total)
    mu11 = float((weights * x * y).sum() / total)
    angle = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
    return float(np.degrees(angle))


def asymmetry_score(image: np.ndarray) -> float:
    """Estimate left-right/up-down imbalance after flipping."""
    arr = to_float_gray(image)
    lr = float(np.mean(np.abs(arr - np.fliplr(arr))))
    ud = float(np.mean(np.abs(arr - np.flipud(arr))))
    return float(np.clip((lr + ud) / (2.0 * (arr.std() + 1e-6)), 0.0, 1.0))


def compute_diffraction_metrics(image: np.ndarray) -> Dict[str, object]:
    """Compute all metrics used by the MVP diffraction diagnosis rules."""
    metrics = compute_basic_metrics(image)
    metrics["fringe_visibility"] = fringe_visibility(image)
    metrics["orientation_deg"] = orientation_estimate_deg(image)
    metrics["asymmetry_score"] = asymmetry_score(image)
    return metrics
