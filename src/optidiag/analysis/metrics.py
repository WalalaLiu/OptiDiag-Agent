"""General image quality metrics used by rule-based diagnosis."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from optidiag.utils.image_io import to_float_gray


def center_of_mass(image: np.ndarray) -> Tuple[float, float]:
    """Return the intensity-weighted center of mass as (x, y)."""
    arr = to_float_gray(image)
    h, w = arr.shape
    total = float(arr.sum())
    if total <= 1e-8:
        return (w / 2.0, h / 2.0)
    y_idx, x_idx = np.indices(arr.shape)
    x = float((x_idx * arr).sum() / total)
    y = float((y_idx * arr).sum() / total)
    return x, y


def laplacian_variance(image: np.ndarray) -> float:
    """Estimate focus sharpness using the variance of a discrete Laplacian."""
    arr = to_float_gray(image)
    lap = (
        -4.0 * arr
        + np.roll(arr, 1, axis=0)
        + np.roll(arr, -1, axis=0)
        + np.roll(arr, 1, axis=1)
        + np.roll(arr, -1, axis=1)
    )
    return float(np.var(lap))


def background_uniformity(image: np.ndarray) -> float:
    """Return 1 for uniform corner background and near 0 for gradients."""
    arr = to_float_gray(image)
    h, w = arr.shape
    pad = max(4, min(h, w) // 8)
    corners = np.concatenate(
        [
            arr[:pad, :pad].ravel(),
            arr[:pad, -pad:].ravel(),
            arr[-pad:, :pad].ravel(),
            arr[-pad:, -pad:].ravel(),
        ]
    )
    corner_mean = float(np.mean(corners))
    corner_std = float(np.std(corners))
    return float(np.clip(1.0 - corner_std / (corner_mean + 0.08), 0.0, 1.0))


def fft_peak_strength(image: np.ndarray) -> float:
    """Estimate the strength of non-DC periodic structure in the FFT."""
    arr = to_float_gray(image)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(arr - np.mean(arr))))
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    radius = max(3, min(h, w) // 30)
    spectrum[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1] = 0.0
    return float(np.max(spectrum) / (np.mean(spectrum) + 1e-6))


def main_lobe_width_estimate(image: np.ndarray) -> float:
    """Estimate central lobe width from the horizontal center profile."""
    arr = to_float_gray(image)
    profile = arr[arr.shape[0] // 2, :]
    if profile.max(initial=0.0) <= 1e-8:
        return 0.0
    threshold = float(profile.max() * 0.5)
    center = len(profile) // 2
    left = center
    while left > 0 and profile[left] >= threshold:
        left -= 1
    right = center
    while right < len(profile) - 1 and profile[right] >= threshold:
        right += 1
    return float(max(0, right - left))


def edge_energy_ratio(image: np.ndarray) -> float:
    """Measure how much bright energy sits near the image border."""
    arr = to_float_gray(image)
    h, w = arr.shape
    pad = max(4, min(h, w) // 14)
    edge_sum = (
        arr[:pad, :].sum()
        + arr[-pad:, :].sum()
        + arr[:, :pad].sum()
        + arr[:, -pad:].sum()
    )
    return float(edge_sum / (arr.sum() + 1e-8))


def compute_basic_metrics(image: np.ndarray) -> Dict[str, object]:
    """Compute general image metrics for diagnostics and explanations."""
    arr = to_float_gray(image)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    p1, p995 = np.percentile(arr, [1, 99.5])
    contrast = float((p995 - p1) / (p995 + p1 + 1e-6))
    saturation = float(np.mean(arr >= 0.98))
    dark_ratio = float(np.mean(arr <= 0.03))
    com_x, com_y = center_of_mass(arr)
    h, w = arr.shape
    offset_x = float(com_x - (w - 1) / 2.0)
    offset_y = float(com_y - (h - 1) / 2.0)

    return {
        "mean_intensity": mean,
        "std_intensity": std,
        "max_intensity": float(np.max(arr)),
        "contrast": contrast,
        "snr_estimate": float(mean / (std + 1e-6)),
        "saturation_ratio": saturation,
        "dark_pixel_ratio": dark_ratio,
        "laplacian_variance": laplacian_variance(arr),
        "center_of_mass": [com_x, com_y],
        "center_offset_px": [offset_x, offset_y],
        "background_uniformity": background_uniformity(arr),
        "fft_peak_strength": fft_peak_strength(arr),
        "main_lobe_width_estimate": main_lobe_width_estimate(arr),
        "edge_energy_ratio": edge_energy_ratio(arr),
    }
