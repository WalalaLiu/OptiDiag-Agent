"""Optional multi-task neural network skeleton for stage-2 training."""

from __future__ import annotations

from optidiag.constants import IMAGE_TYPES, ISSUE_TYPES

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - exercised only without torch installed
    torch = None
    nn = None


class MultiTaskCNN(nn.Module if nn is not None else object):
    """A lightweight CNN with image-type, issue, and quality heads."""

    def __init__(self, num_image_types: int = len(IMAGE_TYPES), num_issues: int = len(ISSUE_TYPES)) -> None:
        if nn is None:
            raise ImportError("Install torch to use MultiTaskCNN.")
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.image_type_head = nn.Linear(96, num_image_types)
        self.issue_head = nn.Linear(96, num_issues)
        self.quality_head = nn.Linear(96, 1)

    def forward(self, x):  # type: ignore[no-untyped-def]
        """Return logits for image type, issue logits, and quality score."""
        features = self.features(x).flatten(1)
        return {
            "image_type_logits": self.image_type_head(features),
            "issue_logits": self.issue_head(features),
            "quality_score": torch.sigmoid(self.quality_head(features)).squeeze(-1),
        }


def torch_available() -> bool:
    """Return whether torch is importable."""
    return torch is not None

