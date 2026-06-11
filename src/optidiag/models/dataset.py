"""Dataset wrapper for synthetic labels.csv files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Union

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

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx]
        image = load_gray_image(self.dataset_dir / row["image_path"])
        issue_scores = json.loads(row["issue_scores"])
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
