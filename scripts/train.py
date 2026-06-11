#!/usr/bin/env python3
"""Stage-2 training entrypoint placeholder.

The MVP does not require model weights. This script is intentionally small so
the later multi-task training stage has a stable command and import surface.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train optional multi-task model.")
    parser.add_argument("--data", default="data/simulated/diffraction_mvp")
    parser.add_argument("--out", default="models/checkpoints/best_model.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, random_split
    except ImportError as exc:
        raise SystemExit("Training requires optional dependencies: torch and torchvision.") from exc

    from optidiag.models.dataset import DiffractionMultiTaskDataset
    from optidiag.models.network import MultiTaskCNN

    args = parse_args()
    dataset = DiffractionMultiTaskDataset(PROJECT_ROOT / args.data)
    train_size = max(1, int(len(dataset) * 0.85))
    val_size = max(0, len(dataset) - train_size)
    train_dataset, _ = random_split(dataset, [train_size, val_size])
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = MultiTaskCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(batch["image"])
            loss = (
                ce_loss(outputs["image_type_logits"], batch["image_type"])
                + bce_loss(outputs["issue_logits"], batch["issues"])
                + mse_loss(outputs["quality_score"], batch["quality_score"])
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        print(f"epoch={epoch + 1} loss={total_loss / max(1, len(loader)):.4f}")

    output_path = PROJECT_ROOT / args.out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, output_path)
    print(f"Saved checkpoint to {output_path}")


if __name__ == "__main__":
    main()

