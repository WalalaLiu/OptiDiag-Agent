"""Artifact sampling and application for MVP dataset generation."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageFilter

from optidiag.constants import ISSUE_TYPES
from optidiag.simulation.noise import (
    add_dark_noise,
    add_gaussian_noise,
    add_poisson_noise,
    apply_background_gradient,
    apply_exposure,
    apply_low_contrast,
)
from optidiag.utils.image_io import array_to_pil, to_float_gray


def _clip_score(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def sample_issue_scores(rng: np.random.Generator) -> Tuple[str, Dict[str, float]]:
    """Sample a primary issue and optional secondary issues."""
    primary_issue = str(rng.choice(ISSUE_TYPES))
    scores = {issue: 0.0 for issue in ISSUE_TYPES}
    scores[primary_issue] = float(rng.uniform(0.5, 0.95))

    if rng.random() < 0.45:
        secondary = str(rng.choice([issue for issue in ISSUE_TYPES if issue != primary_issue]))
        scores[secondary] = float(rng.uniform(0.25, 0.65))
    if rng.random() < 0.2:
        tertiary = str(rng.choice([issue for issue in ISSUE_TYPES if scores[issue] == 0.0]))
        scores[tertiary] = float(rng.uniform(0.15, 0.45))

    return primary_issue, scores


def severity_from_score(score: float) -> str:
    """Map a numeric issue score to a human-readable severity."""
    if score >= 0.72:
        return "high"
    if score >= 0.42:
        return "medium"
    if score > 0.0:
        return "low"
    return "none"


def _rotate(image: np.ndarray, angle_deg: float) -> np.ndarray:
    pil = array_to_pil(image)
    rotated = pil.rotate(angle_deg, resample=Image.Resampling.BICUBIC, fillcolor=0)
    return to_float_gray(rotated)


def _translate(image: np.ndarray, tx: float, ty: float) -> np.ndarray:
    pil = array_to_pil(image)
    shifted = Image.new("L", pil.size, 0)
    shifted.paste(pil, (int(round(tx)), int(round(ty))))
    return to_float_gray(shifted)


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    return to_float_gray(array_to_pil(image).filter(ImageFilter.GaussianBlur(radius=sigma)))


def _crop_incomplete(image: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    size = image.shape[0]
    zoom = 1.0 + 0.18 * strength
    enlarged_size = int(round(size * zoom))
    pil = array_to_pil(image).resize((enlarged_size, enlarged_size), Image.Resampling.BICUBIC)
    max_shift = max(1, int((enlarged_size - size) * (0.55 + 0.4 * strength)))
    dx = int(rng.integers(-max_shift, max_shift + 1))
    dy = int(rng.integers(-max_shift, max_shift + 1))
    left = np.clip((enlarged_size - size) // 2 + dx, 0, enlarged_size - size)
    top = np.clip((enlarged_size - size) // 2 + dy, 0, enlarged_size - size)
    return to_float_gray(pil.crop((left, top, left + size, top + size)))


def apply_artifacts(
    image: np.ndarray,
    issue_scores: Dict[str, float],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float], List[str]]:
    """Apply synthetic acquisition artifacts according to issue scores."""
    output = image.astype(np.float32).copy()
    size = output.shape[0]
    metadata: Dict[str, float] = {
        "noise_type": "none",
        "noise_level": 0.0,
        "blur_sigma": 0.0,
        "rotation_deg": 0.0,
        "translation_x": 0.0,
        "translation_y": 0.0,
        "exposure_factor": 1.0,
        "background_gradient_strength": 0.0,
    }

    low_contrast = _clip_score(issue_scores.get("low_contrast", 0.0))
    if low_contrast > 0:
        output = apply_low_contrast(output, low_contrast)

    blur = _clip_score(issue_scores.get("blur_defocus", 0.0))
    if blur > 0:
        metadata["blur_sigma"] = 0.35 + 2.7 * blur
        output = _blur(output, metadata["blur_sigma"])

    rotation = _clip_score(issue_scores.get("rotation_tilt", 0.0))
    if rotation > 0:
        metadata["rotation_deg"] = float(rng.choice([-1, 1]) * (2.0 + 14.0 * rotation))
        output = _rotate(output, metadata["rotation_deg"])

    misalignment = _clip_score(issue_scores.get("misalignment", 0.0))
    if misalignment > 0:
        magnitude = size * (0.03 + 0.18 * misalignment)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        metadata["translation_x"] = float(np.cos(angle) * magnitude)
        metadata["translation_y"] = float(np.sin(angle) * magnitude)
        output = _translate(output, metadata["translation_x"], metadata["translation_y"])

    cropping = _clip_score(issue_scores.get("cropping_incomplete", 0.0))
    if cropping > 0:
        output = _crop_incomplete(output, cropping, rng)

    gradient = _clip_score(issue_scores.get("background_gradient", 0.0))
    if gradient > 0:
        metadata["background_gradient_strength"] = gradient
        output = apply_background_gradient(output, gradient, rng)

    over = _clip_score(issue_scores.get("over_exposure", 0.0))
    under = _clip_score(issue_scores.get("under_exposure", 0.0))
    if over > 0:
        metadata["exposure_factor"] *= 1.0 + 2.4 * over
    if under > 0:
        metadata["exposure_factor"] *= max(0.12, 1.0 - 0.78 * under)
    if over > 0 or under > 0:
        output = apply_exposure(output, metadata["exposure_factor"])

    noise_candidates = [
        ("gaussian_noise", _clip_score(issue_scores.get("gaussian_noise", 0.0))),
        ("poisson_noise", _clip_score(issue_scores.get("poisson_noise", 0.0))),
        ("dark_noise", _clip_score(issue_scores.get("dark_noise", 0.0))),
    ]
    active_noise = max(noise_candidates, key=lambda item: item[1])
    if active_noise[1] > 0:
        metadata["noise_type"] = active_noise[0]
        metadata["noise_level"] = active_noise[1]
        if active_noise[0] == "gaussian_noise":
            output = add_gaussian_noise(output, 0.015 + 0.09 * active_noise[1], rng)
        elif active_noise[0] == "poisson_noise":
            output = add_poisson_noise(output, active_noise[1], rng)
        else:
            output = add_dark_noise(output, active_noise[1], rng)

    active_issues = [issue for issue, score in issue_scores.items() if score >= 0.2]
    return np.clip(output, 0.0, 1.0), metadata, active_issues

