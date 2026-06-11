"""Rule-based fallback diagnosis for diffraction images."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from optidiag.analysis.diffraction_metrics import compute_diffraction_metrics
from optidiag.reasoning.report_generator import build_causes_and_suggestions, build_diagnosis_sentence
from optidiag.utils.image_io import to_float_gray


def _severity(score: float) -> str:
    if score >= 0.72:
        return "high"
    if score >= 0.42:
        return "medium"
    return "low"


def _add_issue(issues: List[Dict[str, object]], issue_type: str, score: float) -> None:
    score = float(np.clip(score, 0.0, 1.0))
    if score <= 0.18:
        return
    issues.append({"type": issue_type, "severity": _severity(score), "score": round(score, 3)})


def diagnose_from_metrics(
    metrics: Dict[str, object],
    image_type: str = "unknown_or_estimated",
    confidence: float = 0.0,
) -> Dict[str, object]:
    """Convert metrics to a diagnosis response with Chinese suggestions."""
    issues: List[Dict[str, object]] = []
    mean = float(metrics["mean_intensity"])
    std = float(metrics["std_intensity"])
    max_intensity = float(metrics.get("max_intensity", 0.0))
    contrast = float(metrics["contrast"])
    snr = float(metrics["snr_estimate"])
    saturation = float(metrics["saturation_ratio"])
    dark = float(metrics["dark_pixel_ratio"])
    lap_var = float(metrics["laplacian_variance"])
    uniformity = float(metrics["background_uniformity"])
    background_gradient = float(metrics.get("background_gradient_estimate", 0.0))
    fringe = float(metrics["fringe_visibility"])
    edge_energy = float(metrics.get("edge_energy_ratio", 0.0))
    asymmetry = float(metrics.get("asymmetry_score", 0.0))
    orientation = abs(float(metrics.get("orientation_deg", 0.0)))
    offset_x, offset_y = metrics["center_offset_px"]
    offset_mag = float(np.hypot(float(offset_x), float(offset_y)))

    _add_issue(issues, "over_exposure", saturation / 0.006)
    _add_issue(issues, "under_exposure", max(0.0, (0.50 - max_intensity) / 0.42) * max(0.0, (0.13 - mean) / 0.13))
    _add_issue(issues, "blur_defocus", (0.0018 - lap_var) / 0.0018)
    background_score = (0.90 - uniformity) / 0.30
    background_score = max(background_score, background_gradient / 0.28)
    _add_issue(issues, "background_gradient", background_score)
    _add_issue(issues, "misalignment", (offset_mag / 32.0) * max(0.25, 1.0 - min(1.0, background_score) * 0.55))
    fringe_score = (
        (0.28 - fringe) / 0.28
        if contrast < 0.55 and max_intensity > 0.45 and edge_energy < 0.25
        else 0.0
    )
    _add_issue(issues, "low_contrast", max((0.42 - contrast) / 0.42, fringe_score))
    _add_issue(issues, "cropping_incomplete", (edge_energy - 0.18) / 0.25)
    _add_issue(issues, "rotation_tilt", max(0.0, min(orientation, 90.0 - orientation) - 8.0) / 24.0)

    high_frequency_noise = max(0.0, std - 0.12) * max(0.0, lap_var - 0.01) * 45.0
    _add_issue(issues, "gaussian_noise", high_frequency_noise)
    _add_issue(issues, "poisson_noise", max(0.0, 0.34 - mean) * max(0.0, std - 0.11) * 4.0)
    _add_issue(issues, "dark_noise", max(0.0, dark - 0.75) * max(0.0, std - 0.09) * max(0.0, 0.98 - max_intensity) * 2.0)

    # Prefer clearer, higher-confidence issues in the final response.
    dedup: Dict[str, Dict[str, object]] = {}
    for issue in sorted(issues, key=lambda item: float(item["score"]), reverse=True):
        dedup.setdefault(str(issue["type"]), issue)
    issues = list(dedup.values())[:5]

    diagnosis = build_diagnosis_sentence(issues)
    possible_causes, suggestions = build_causes_and_suggestions(issues)
    need_reacquire = any(issue["severity"] in {"medium", "high"} for issue in issues)

    return {
        "image_type": image_type,
        "confidence": float(confidence),
        "issues": issues,
        "metrics": metrics,
        "diagnosis": diagnosis,
        "possible_causes": possible_causes,
        "suggestions": suggestions,
        "need_reacquire": bool(need_reacquire),
        "model_available": False,
        "fallback": "rule_based",
    }


def analyze_image_rule_based(image: object, image_type: Optional[str] = None) -> Dict[str, object]:
    """Run rule-based diagnosis directly on an image-like object."""
    arr = to_float_gray(image)
    metrics = compute_diffraction_metrics(arr)
    return diagnose_from_metrics(metrics, image_type=image_type or "unknown_or_estimated", confidence=0.0)
