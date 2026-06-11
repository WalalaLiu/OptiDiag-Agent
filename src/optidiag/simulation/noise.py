"""Synthetic noise and exposure operators."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def add_gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian sensor noise."""
    return np.clip(image + rng.normal(0.0, sigma, size=image.shape), 0.0, 1.0)


def add_poisson_noise(image: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Apply photon-counting noise; larger strength means fewer effective photons."""
    photons = max(8.0, 90.0 * (1.0 - strength) + 12.0)
    noisy = rng.poisson(np.clip(image, 0.0, 1.0) * photons) / photons
    return np.clip(noisy, 0.0, 1.0)


def add_dark_noise(image: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Add dark-current background and sparse hot pixels."""
    dark_current = rng.exponential(scale=0.025 * strength, size=image.shape)
    hot_pixel_mask = rng.random(image.shape) < (0.001 + 0.012 * strength)
    hot_pixels = hot_pixel_mask.astype(np.float32) * rng.uniform(0.2, 0.8, size=image.shape)
    return np.clip(image + dark_current + hot_pixels, 0.0, 1.0)


def apply_background_gradient(image: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Overlay a smooth background illumination gradient."""
    h, w = image.shape
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    gradient = np.cos(angle) * xx + np.sin(angle) * yy
    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-8)
    return np.clip(image + strength * 0.45 * gradient, 0.0, 1.0)


def apply_exposure(image: np.ndarray, factor: float) -> np.ndarray:
    """Scale image exposure and clip to the sensor range."""
    return np.clip(image * factor, 0.0, 1.0)


def apply_low_contrast(image: np.ndarray, strength: float) -> np.ndarray:
    """Compress contrast around the image mean."""
    mean = float(image.mean())
    factor = max(0.15, 1.0 - 0.75 * strength)
    return np.clip((image - mean) * factor + mean, 0.0, 1.0)


def estimate_snr(image: np.ndarray) -> float:
    """Estimate a simple signal-to-noise ratio from image moments."""
    mean = float(np.mean(image))
    std = float(np.std(image))
    return mean / (std + 1e-6)


def percentile_contrast(image: np.ndarray) -> float:
    """Compute robust contrast from the 5th and 95th percentiles."""
    p1, p999 = np.percentile(image, [1, 99.9])
    return float((p999 - p1) / (p999 + p1 + 1e-6))


def exposure_stats(image: np.ndarray) -> Tuple[float, float]:
    """Return saturation and dark-pixel ratios."""
    return float(np.mean(image >= 0.98)), float(np.mean(image <= 0.03))
