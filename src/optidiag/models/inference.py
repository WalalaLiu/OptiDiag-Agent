"""Optional model inference wrapper with a safe rule-based fallback path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import numpy as np

from optidiag.constants import IMAGE_TYPES, ISSUE_TYPES
from optidiag.models.checkpoint import load_torch_checkpoint
from optidiag.models.network import MultiTaskCNN, torch_available
from optidiag.utils.image_io import to_float_gray

if torch_available():
    import torch
else:  # pragma: no cover
    torch = None


class ModelInference:
    """Load an optional checkpoint and return predictions when available."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "models/checkpoints/best_model.pt",
        issue_threshold: float = 0.5,
        issue_thresholds: Optional[Union[str, Path, Mapping[str, float]]] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.issue_threshold = float(issue_threshold)
        self.issue_thresholds = self._load_issue_thresholds(issue_thresholds)
        self.model = None
        self.available = False
        if torch is None or not self.checkpoint_path.exists():
            return
        try:
            self.model = MultiTaskCNN()
            checkpoint = load_torch_checkpoint(self.checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            self.available = True
        except Exception:
            self.model = None
            self.available = False

    def _load_issue_thresholds(self, issue_thresholds: Optional[Union[str, Path, Mapping[str, float]]]) -> Dict[str, float]:
        """Load optional per-issue thresholds from a mapping or JSON file."""
        if issue_thresholds is None:
            return {}
        if isinstance(issue_thresholds, Mapping):
            return {str(issue): float(value) for issue, value in issue_thresholds.items() if str(issue) in ISSUE_TYPES}

        path = Path(issue_thresholds)
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError, TypeError):
            return {}

        if "per_class_thresholds" in payload:
            payload = payload["per_class_thresholds"]
        thresholds: Dict[str, float] = {}
        for issue, value in payload.items():
            if issue not in ISSUE_TYPES:
                continue
            if isinstance(value, Mapping):
                value = value.get("best_threshold", self.issue_threshold)
            thresholds[issue] = float(value)
        return thresholds

    def _threshold_for_issue(self, issue: str) -> float:
        """Return per-issue threshold or the configured global default."""
        return float(self.issue_thresholds.get(issue, self.issue_threshold))

    def predict(self, image: object) -> Optional[Dict[str, object]]:
        """Return model predictions, or None when no checkpoint is available."""
        try:
            if not self.available or self.model is None or torch is None:
                return None
            arr = to_float_gray(image)
            tensor = torch.from_numpy(arr[None, None, :, :].astype(np.float32))
            with torch.no_grad():
                outputs = self.model(tensor)
                image_probs = torch.softmax(outputs["image_type_logits"], dim=1)[0]
                issue_probs = torch.sigmoid(outputs["issue_logits"])[0]
                image_idx = int(torch.argmax(image_probs).item())
        except Exception:
            return None

        issues = []
        for issue, prob in zip(ISSUE_TYPES, issue_probs.tolist()):
            if prob >= self._threshold_for_issue(issue):
                severity = "high" if prob >= 0.72 else "medium" if prob >= 0.42 else "low"
                issues.append({"type": issue, "severity": severity, "score": round(float(prob), 3)})

        return {
            "image_type": IMAGE_TYPES[image_idx],
            "confidence": float(image_probs[image_idx].item()),
            "issues": issues,
            "quality_score": float(outputs["quality_score"][0].item()),
        }
