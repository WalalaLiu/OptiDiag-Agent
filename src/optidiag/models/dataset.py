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
    """Load synthetic diffraction images and multi-task labels."""

    def __init__(self, dataset_dir: Union[str, Path]) -> None:
        if torch is None:
            raise ImportError("Install torch to use DiffractionMultiTaskDataset.")
        self.dataset_dir = Path(dataset_dir)
        labels_path = self.dataset_dir / "labels.csv"
        with labels_path.open("r", encoding="utf-8") as file:
            self.rows: List[Dict[str, str]] = list(csv.DictReader(file))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx]
        image = load_gray_image(self.dataset_dir / row["image_path"])
        issue_scores = json.loads(row["issue_scores"])
        issue_vector = np.array([float(issue_scores.get(issue, 0.0) > 0.2) for issue in ISSUE_TYPES], dtype=np.float32)
        image_type_idx = IMAGE_TYPES.index(row["image_type"])

        return {
            "image": torch.from_numpy(image[None, :, :].astype(np.float32)),
            "image_type": torch.tensor(image_type_idx, dtype=torch.long),
            "issues": torch.from_numpy(issue_vector),
            "quality_score": torch.tensor(float(row["quality_score"]), dtype=torch.float32),
        }
