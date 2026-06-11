#!/usr/bin/env python3
"""Train a small multi-task CNN on synthetic diffraction labels.csv data."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train optional multi-task model.")
    parser.add_argument("--data-dir", "--data", dest="data_dir", default="data/simulated/diffraction_mvp")
    parser.add_argument("--checkpoint", "--out", dest="checkpoint", default="models/checkpoints/best_model.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _prepare_device(requested: str):  # type: ignore[no-untyped-def]
    """Return a torch device, falling back to CPU when needed."""
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA but it is unavailable; falling back to CPU.")
        requested = "cpu"
    if requested == "mps" and not getattr(torch.backends, "mps", None):
        print("Requested MPS but it is unavailable; falling back to CPU.")
        requested = "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        print("Requested MPS but it is unavailable; falling back to CPU.")
        requested = "cpu"
    return torch.device(requested)


def _compute_metrics(outputs: Dict[str, object], batch: Dict[str, object]) -> Dict[str, float]:
    """Compute lightweight batch metrics."""
    import torch

    image_preds = torch.argmax(outputs["image_type_logits"], dim=1)
    image_acc = (image_preds == batch["image_type"]).float().mean().item()

    issue_pred = (torch.sigmoid(outputs["issue_logits"]) >= 0.5).float()
    issue_true = batch["issues"].float()
    tp = (issue_pred * issue_true).sum().item()
    fp = (issue_pred * (1.0 - issue_true)).sum().item()
    fn = ((1.0 - issue_pred) * issue_true).sum().item()
    issue_f1 = (2.0 * tp) / max(1e-8, 2.0 * tp + fp + fn)

    quality_mae = torch.mean(torch.abs(outputs["quality_score"] - batch["quality_score"])).item()
    return {"image_acc": image_acc, "issue_f1": issue_f1, "quality_mae": quality_mae}


def _run_epoch(model, loader, criterion, optimizer, device, train: bool) -> Dict[str, float]:  # type: ignore[no-untyped-def]
    """Run one training or validation epoch."""
    import torch

    ce_loss, bce_loss, mse_loss = criterion
    model.train(mode=train)
    total_loss = 0.0
    total_image_acc = 0.0
    total_issue_f1 = 0.0
    total_quality_mae = 0.0
    batches = 0

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            outputs = model(batch["image"])
            loss = (
                ce_loss(outputs["image_type_logits"], batch["image_type"])
                + bce_loss(outputs["issue_logits"], batch["issues"])
                + mse_loss(outputs["quality_score"], batch["quality_score"])
            )
            if train:
                loss.backward()
                optimizer.step()

        metrics = _compute_metrics(outputs, batch)
        total_loss += float(loss.item())
        total_image_acc += metrics["image_acc"]
        total_issue_f1 += metrics["issue_f1"]
        total_quality_mae += metrics["quality_mae"]
        batches += 1

    return {
        "loss": total_loss / max(1, batches),
        "image_acc": total_image_acc / max(1, batches),
        "issue_f1": total_issue_f1 / max(1, batches),
        "quality_mae": total_quality_mae / max(1, batches),
    }


def main() -> None:
    """Train and save the best validation checkpoint."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, random_split
    except ImportError as exc:
        raise SystemExit("Training requires optional dependency: torch.") from exc

    from optidiag.constants import IMAGE_TYPES, ISSUE_TYPES
    from optidiag.models.dataset import DiffractionMultiTaskDataset
    from optidiag.models.network import MultiTaskCNN

    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _prepare_device(args.device)

    dataset = DiffractionMultiTaskDataset(PROJECT_ROOT / args.data_dir)
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = max(1, len(dataset) - val_size)
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = MultiTaskCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = (nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss(), nn.MSELoss())

    print(f"dataset_size={len(dataset)} train_size={train_size} val_size={val_size}")
    print(f"device={device} epochs={args.epochs} batch_size={args.batch_size}")

    best_val_loss = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            "epoch={epoch} "
            "train_loss={train_loss:.4f} train_image_acc={train_acc:.3f} train_issue_f1={train_f1:.3f} "
            "val_loss={val_loss:.4f} val_image_acc={val_acc:.3f} val_issue_f1={val_f1:.3f} val_quality_mae={val_mae:.3f}".format(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_acc=train_metrics["image_acc"],
                train_f1=train_metrics["issue_f1"],
                val_loss=val_metrics["loss"],
                val_acc=val_metrics["image_acc"],
                val_f1=val_metrics["issue_f1"],
                val_mae=val_metrics["quality_mae"],
            )
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            output_path = PROJECT_ROOT / args.checkpoint
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_types": IMAGE_TYPES,
                    "issue_types": ISSUE_TYPES,
                    "train_args": vars(args),
                    "best_val_loss": best_val_loss,
                    "history": history,
                },
                output_path,
            )
            print(f"saved_checkpoint={output_path}")

    metrics_path = (PROJECT_ROOT / "runs" / "train_small_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump({"history": history, "best_val_loss": best_val_loss}, file, indent=2)
    print(f"metrics_path={metrics_path}")


if __name__ == "__main__":
    main()
