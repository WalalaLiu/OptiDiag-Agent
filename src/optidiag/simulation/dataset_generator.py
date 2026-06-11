"""Synthetic dataset generator for the OptiDiag MVP."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np

from optidiag.analysis.diffraction_metrics import compute_diffraction_metrics
from optidiag.constants import IMAGE_TYPES, SUGGESTION_LABELS
from optidiag.simulation.augmentation import apply_artifacts, sample_issue_scores, severity_from_score
from optidiag.simulation.diffraction import DiffractionSimulator
from optidiag.utils.image_io import save_gray_png


LABEL_COLUMNS = [
    "image_path",
    "json_path",
    "image_type",
    "primary_issue",
    "issues",
    "issue_scores",
    "severity",
    "noise_type",
    "noise_level",
    "blur_sigma",
    "rotation_deg",
    "translation_x",
    "translation_y",
    "exposure_factor",
    "background_gradient_strength",
    "contrast",
    "snr_estimate",
    "saturation_ratio",
    "center_offset_x",
    "center_offset_y",
    "fringe_visibility",
    "quality_score",
    "suggestion_label",
]


@dataclass
class DatasetGenerator:
    """Generate PNG, JSON metadata, and labels.csv for synthetic diffraction data."""

    image_size: int = 256
    grid_size: int = 512
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.simulator = DiffractionSimulator(image_size=self.image_size, grid_size=self.grid_size)

    def _quality_score(self, metrics: Dict[str, object], primary_score: float) -> float:
        saturation_penalty = float(metrics["saturation_ratio"]) * 1.4
        contrast_penalty = max(0.0, 0.45 - float(metrics["contrast"]))
        offset = metrics["center_offset_px"]
        offset_penalty = min(0.5, (abs(float(offset[0])) + abs(float(offset[1]))) / (2.0 * self.image_size))
        quality = 1.0 - max(primary_score * 0.55, saturation_penalty, contrast_penalty, offset_penalty)
        return float(np.clip(quality, 0.0, 1.0))

    def _relative(self, path: Path, root: Path) -> str:
        return str(path.relative_to(root)).replace("\\", "/")

    def generate(
        self,
        out_dir: Union[str, Path],
        num_images: int,
        image_types: Optional[Iterable[str]] = None,
    ) -> Path:
        """Generate a dataset and return the labels.csv path."""
        output_root = Path(out_dir)
        images_dir = output_root / "images"
        meta_dir = output_root / "metadata"
        images_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        selected_types: List[str] = list(image_types or IMAGE_TYPES)
        labels_path = output_root / "labels.csv"

        with labels_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=LABEL_COLUMNS)
            writer.writeheader()

            for idx in range(num_images):
                image_type = selected_types[idx % len(selected_types)]
                clean, aperture_params = self.simulator.render(image_type, rng=self.rng)
                primary_issue, issue_scores = sample_issue_scores(self.rng)
                artifact_image, artifact_meta, active_issues = apply_artifacts(clean, issue_scores, self.rng)
                metrics = compute_diffraction_metrics(artifact_image)

                primary_score = float(issue_scores[primary_issue])
                severity = severity_from_score(primary_score)
                quality_score = self._quality_score(metrics, primary_score)

                stem = f"{idx:05d}_{image_type}_{primary_issue}"
                image_path = images_dir / f"{stem}.png"
                json_path = meta_dir / f"{stem}.json"
                save_gray_png(artifact_image, image_path)

                metadata = {
                    "image_type": image_type,
                    "primary_issue": primary_issue,
                    "issues": active_issues,
                    "issue_scores": issue_scores,
                    "severity": severity,
                    "aperture_params": aperture_params,
                    "artifact_params": artifact_meta,
                    "metrics": metrics,
                    "quality_score": quality_score,
                    "suggestion_label": SUGGESTION_LABELS[primary_issue],
                    "simulation": {
                        "image_size": self.image_size,
                        "grid_size": self.grid_size,
                        "wavelength_nm": self.simulator.wavelength_nm,
                        "focal_length_mm": self.simulator.focal_length_mm,
                    },
                }
                with json_path.open("w", encoding="utf-8") as json_file:
                    json.dump(metadata, json_file, ensure_ascii=False, indent=2)

                center_offset = metrics["center_offset_px"]
                row = {
                    "image_path": self._relative(image_path, output_root),
                    "json_path": self._relative(json_path, output_root),
                    "image_type": image_type,
                    "primary_issue": primary_issue,
                    "issues": json.dumps(active_issues, ensure_ascii=False),
                    "issue_scores": json.dumps(issue_scores, ensure_ascii=False),
                    "severity": severity,
                    "noise_type": artifact_meta["noise_type"],
                    "noise_level": artifact_meta["noise_level"],
                    "blur_sigma": artifact_meta["blur_sigma"],
                    "rotation_deg": artifact_meta["rotation_deg"],
                    "translation_x": artifact_meta["translation_x"],
                    "translation_y": artifact_meta["translation_y"],
                    "exposure_factor": artifact_meta["exposure_factor"],
                    "background_gradient_strength": artifact_meta["background_gradient_strength"],
                    "contrast": metrics["contrast"],
                    "snr_estimate": metrics["snr_estimate"],
                    "saturation_ratio": metrics["saturation_ratio"],
                    "center_offset_x": center_offset[0],
                    "center_offset_y": center_offset[1],
                    "fringe_visibility": metrics["fringe_visibility"],
                    "quality_score": quality_score,
                    "suggestion_label": SUGGESTION_LABELS[primary_issue],
                }
                writer.writerow(row)

        return labels_path
