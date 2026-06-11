"""Dataset wrapper for synthetic labels.csv files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np

from optidiag.constants import IMAGE_TYPES, ISSUE_TYPES
from optidiag.utils.image_io import load_gray_image

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object


class DiffractionMultiTaskDataset(Dataset):
    """Load synthetic diffraction images and multi-task labels from labels.csv."""

    image_type_names = IMAGE_TYPES
    issue_names = ISSUE_TYPES

    def __init__(self, dataset_dir: Union[str, Path], labels_file: str = "labels.csv") -> None:
        if torch is None:
            raise ImportError("Install torch to use DiffractionMultiTaskDataset.")
        self.dataset_dir = Path(dataset_dir)
        labels_path = self.dataset_dir / labels_file
        if not labels_path.exists():
            raise FileNotFoundError(f"labels.csv not found: {labels_path}")
        with labels_path.open("r", encoding="utf-8") as file:
            self.rows: List[Dict[str, str]] = list(csv.DictReader(file))
        if not self.rows:
            raise ValueError(f"No rows found in labels file: {labels_path}")
        self._validate_rows()

    def __len__(self) -> int:
        return len(self.rows)

    def _validate_rows(self) -> None:
        """Validate required labels and keep failures early and explicit."""
        required = {"image_path", "image_type", "primary_issue", "issue_scores", "quality_score"}
        missing = required - set(self.rows[0].keys())
        if missing:
            raise ValueError(f"Missing labels.csv columns: {sorted(missing)}")
        unknown_images = sorted({row["image_type"] for row in self.rows if row["image_type"] not in IMAGE_TYPES})
        unknown_issues = sorted({row["primary_issue"] for row in self.rows if row["primary_issue"] not in ISSUE_TYPES})
        if unknown_images:
            raise ValueError(f"Unknown image_type labels: {unknown_images}")
        if unknown_issues:
            raise ValueError(f"Unknown primary_issue labels: {unknown_issues}")

    def _issue_scores(self, row: Dict[str, str]) -> Dict[str, float]:
        """Parse issue_scores into the fixed ISSUE_TYPES order."""
        scores = json.loads(row["issue_scores"])
        return {issue: float(scores.get(issue, 0.0)) for issue in ISSUE_TYPES}

    def issue_matrix(self, indices: Optional[Iterable[int]] = None) -> np.ndarray:
        """Return a multi-hot issue matrix in fixed ISSUE_TYPES order."""
        selected_indices = list(indices) if indices is not None else list(range(len(self.rows)))
        matrix = []
        for idx in selected_indices:
            scores = self._issue_scores(self.rows[int(idx)])
            matrix.append([float(scores.get(issue, 0.0) > 0.2) for issue in ISSUE_TYPES])
        return np.asarray(matrix, dtype=np.float32)

    def primary_issue_indices(self, indices: Optional[Iterable[int]] = None) -> np.ndarray:
        """Return primary issue indices in fixed ISSUE_TYPES order."""
        selected_indices = list(indices) if indices is not None else list(range(len(self.rows)))
        return np.asarray([ISSUE_TYPES.index(self.rows[int(idx)]["primary_issue"]) for idx in selected_indices], dtype=np.int64)

    def image_type_indices(self, indices: Optional[Iterable[int]] = None) -> np.ndarray:
        """Return image type indices in fixed IMAGE_TYPES order."""
        selected_indices = list(indices) if indices is not None else list(range(len(self.rows)))
        return np.asarray([IMAGE_TYPES.index(self.rows[int(idx)]["image_type"]) for idx in selected_indices], dtype=np.int64)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx]
        image = load_gray_image(self.dataset_dir / row["image_path"])
        issue_scores = self._issue_scores(row)
        issue_vector = np.array([float(issue_scores.get(issue, 0.0) > 0.2) for issue in ISSUE_TYPES], dtype=np.float32)
        issue_score_vector = np.array([float(issue_scores.get(issue, 0.0)) for issue in ISSUE_TYPES], dtype=np.float32)
        image_type_idx = IMAGE_TYPES.index(row["image_type"])
        primary_issue_idx = ISSUE_TYPES.index(row["primary_issue"]) if row["primary_issue"] in ISSUE_TYPES else -1

        return {
            "image": torch.from_numpy(image[None, :, :].astype(np.float32)),
            "image_type": torch.tensor(image_type_idx, dtype=torch.long),
            "issues": torch.from_numpy(issue_vector),
            "issue_scores": torch.from_numpy(issue_score_vector),
            "quality_score": torch.tensor(float(row["quality_score"]), dtype=torch.float32),
            "primary_issue": torch.tensor(primary_issue_idx, dtype=torch.long),
        }
