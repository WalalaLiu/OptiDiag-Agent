"""Optional model inference wrapper with a safe rule-based fallback path."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from optidiag.constants import IMAGE_TYPES, ISSUE_TYPES
from optidiag.models.network import MultiTaskCNN, torch_available
from optidiag.utils.image_io import to_float_gray

if torch_available():
    import torch
else:  # pragma: no cover
    torch = None


class ModelInference:
    """Load an optional checkpoint and return predictions when available."""

    def __init__(self, checkpoint_path: Union[str, Path] = "models/checkpoints/best_model.pt") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.model = None
        self.available = False
        if torch is None or not self.checkpoint_path.exists():
            return
        self.model = MultiTaskCNN()
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.available = True

    def predict(self, image: object) -> Optional[Dict[str, object]]:
        """Return model predictions, or None when no checkpoint is available."""
        if not self.available or self.model is None or torch is None:
            return None
        arr = to_float_gray(image)
        tensor = torch.from_numpy(arr[None, None, :, :].astype(np.float32))
        with torch.no_grad():
            outputs = self.model(tensor)
            image_probs = torch.softmax(outputs["image_type_logits"], dim=1)[0]
            issue_probs = torch.sigmoid(outputs["issue_logits"])[0]
            image_idx = int(torch.argmax(image_probs).item())

        issues = []
        for issue, prob in zip(ISSUE_TYPES, issue_probs.tolist()):
            if prob >= 0.25:
                severity = "high" if prob >= 0.72 else "medium" if prob >= 0.42 else "low"
                issues.append({"type": issue, "severity": severity, "score": round(float(prob), 3)})

        return {
            "image_type": IMAGE_TYPES[image_idx],
            "confidence": float(image_probs[image_idx].item()),
            "issues": issues,
            "quality_score": float(outputs["quality_score"][0].item()),
        }
