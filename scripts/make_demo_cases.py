#!/usr/bin/env python3
"""Generate deterministic report/demo cases for Phase 1.5."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from optidiag.constants import ISSUE_TYPES  # noqa: E402
from optidiag.reasoning.rules import analyze_image_rule_based  # noqa: E402
from optidiag.simulation.augmentation import apply_artifacts  # noqa: E402
from optidiag.simulation.diffraction import DiffractionSimulator  # noqa: E402
from optidiag.utils.image_io import array_to_pil, save_gray_png, to_float_gray  # noqa: E402
from optidiag.utils.visualization import create_demo_visualization  # noqa: E402


DEMO_CASES = [
    {"case_id": "01_over_exposure", "image_type": "single_slit", "issue": "over_exposure", "score": 0.88},
    {"case_id": "02_under_exposure", "image_type": "double_slit", "issue": "under_exposure", "score": 0.92},
    {"case_id": "03_blur_defocus", "image_type": "triple_slit", "issue": "blur_defocus", "score": 0.9},
    {"case_id": "04_misalignment", "image_type": "double_slit", "issue": "misalignment", "score": 0.9},
    {"case_id": "05_background_gradient", "image_type": "circular_aperture", "issue": "background_gradient", "score": 0.86},
    {"case_id": "06_gaussian_noise", "image_type": "double_circular_aperture", "issue": "gaussian_noise", "score": 0.95},
    {"case_id": "07_low_contrast", "image_type": "double_slit", "issue": "low_contrast", "score": 0.92},
    {"case_id": "08_cropping_incomplete", "image_type": "circular_aperture", "issue": "cropping_incomplete", "score": 0.9},
]

SUMMARY_COLUMNS = [
    "case_id",
    "image_type",
    "expected_issue",
    "detected_issues",
    "top_issue",
    "need_reacquire",
    "image_path",
    "result_path",
    "visualization_path",
    "diagnosis",
]


def issue_scores(primary_issue: str, score: float) -> Dict[str, float]:
    """Build a single-primary issue score dictionary."""
    scores = {issue: 0.0 for issue in ISSUE_TYPES}
    scores[primary_issue] = score
    return scores


def add_demo_background_gradient(image: np.ndarray, strength: float = 0.7) -> np.ndarray:
    """Create a clear smooth background-gradient demo image."""
    h, w = image.shape
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    gradient = 0.65 * xx + 0.35 * yy
    return np.clip(image * 0.28 + gradient * strength, 0.0, 0.92)


def add_demo_gaussian_noise(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Create a high-frequency Gaussian noise demo without heavy saturation."""
    noise = rng.normal(0.0, 0.16, size=image.shape)
    return np.clip(image * 0.55 + noise + 0.08, 0.0, 0.9)


def add_demo_low_contrast(image: np.ndarray) -> np.ndarray:
    """Create a visible low-contrast demo while avoiding underexposure."""
    return np.clip(image * 0.18 + 0.30, 0.0, 0.55)


def add_demo_cropping(image: np.ndarray) -> np.ndarray:
    """Create an incomplete field-of-view demo through zooming and offset crop."""
    size = image.shape[0]
    enlarged = array_to_pil(image).resize((int(size * 1.45), int(size * 1.45)), resample=2)
    crop = enlarged.crop((int(size * 0.40), int(size * 0.18), int(size * 1.40), int(size * 1.18)))
    return np.clip(to_float_gray(crop) * 0.95, 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 1.5 demo cases.")
    parser.add_argument("--out", default="outputs/demo_cases", help="Output directory.")
    parser.add_argument("--image-size", type=int, default=256, help="Rendered demo image size.")
    parser.add_argument("--grid-size", type=int, default=512, help="Simulation grid size.")
    parser.add_argument("--seed", type=int, default=20260611, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = PROJECT_ROOT / args.out
    images_dir = output_root / "images"
    results_dir = output_root / "results"
    visualizations_dir = output_root / "visualizations"
    for directory in (images_dir, results_dir, visualizations_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    simulator = DiffractionSimulator(image_size=args.image_size, grid_size=args.grid_size)
    rows: List[Dict[str, object]] = []

    for idx, case in enumerate(DEMO_CASES):
        case_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        clean, aperture_params = simulator.render(case["image_type"], rng=case_rng)
        artifact_image, artifact_meta, active_issues = apply_artifacts(
            clean,
            issue_scores(str(case["issue"]), float(case["score"])),
            case_rng,
        )
        if case["issue"] == "background_gradient":
            artifact_image = add_demo_background_gradient(clean)
        elif case["issue"] == "gaussian_noise":
            artifact_image = add_demo_gaussian_noise(clean, case_rng)
        elif case["issue"] == "low_contrast":
            artifact_image = add_demo_low_contrast(clean)
        elif case["issue"] == "cropping_incomplete":
            artifact_image = add_demo_cropping(clean)

        image_path = images_dir / f"{case['case_id']}.png"
        result_path = results_dir / f"{case['case_id']}.json"
        visualization_path = visualizations_dir / f"{case['case_id']}_report.png"
        save_gray_png(artifact_image, image_path)

        analysis = analyze_image_rule_based(artifact_image, image_type=str(case["image_type"]))
        result_payload = {
            "case": case,
            "active_issues": active_issues,
            "aperture_params": aperture_params,
            "artifact_params": artifact_meta,
            "analysis": analysis,
        }
        with result_path.open("w", encoding="utf-8") as file:
            json.dump(result_payload, file, ensure_ascii=False, indent=2)

        create_demo_visualization(
            artifact_image,
            analysis,
            visualization_path,
            title=f"{case['case_id']}: {case['issue']}",
            expected_issue=str(case["issue"]),
        )

        detected = [item["type"] for item in analysis["issues"]]
        top_issue = detected[0] if detected else ""
        rows.append(
            {
                "case_id": case["case_id"],
                "image_type": case["image_type"],
                "expected_issue": case["issue"],
                "detected_issues": ";".join(detected),
                "top_issue": top_issue,
                "need_reacquire": analysis["need_reacquire"],
                "image_path": str(image_path.relative_to(output_root)),
                "result_path": str(result_path.relative_to(output_root)),
                "visualization_path": str(visualization_path.relative_to(output_root)),
                "diagnosis": analysis["diagnosis"],
            }
        )

    summary_path = output_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} demo cases")
    print(f"Output: {output_root}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
