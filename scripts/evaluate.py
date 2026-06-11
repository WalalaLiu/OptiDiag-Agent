#!/usr/bin/env python3
"""Evaluate the optional multi-task CNN on a labels.csv dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained multi-task checkpoint.")
    parser.add_argument("--data-dir", "--data", dest="data_dir", default="data/simulated/diffraction_mvp")
    parser.add_argument("--checkpoint", default="models/checkpoints/best_model.pt")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _prepare_device(requested: str):  # type: ignore[no-untyped-def]
    """Return a torch device, falling back to CPU when needed."""
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA but it is unavailable; falling back to CPU.")
        requested = "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        print("Requested MPS but it is unavailable; falling back to CPU.")
        requested = "cpu"
    return torch.device(requested)


def main() -> None:
    """Evaluate image type accuracy, issue F1, and quality MAE."""
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("Evaluation requires optional dependency: torch.") from exc

    from optidiag.constants import ISSUE_TYPES
    from optidiag.models.dataset import DiffractionMultiTaskDataset
    from optidiag.models.network import MultiTaskCNN

    args = parse_args()
    device = _prepare_device(args.device)
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    dataset = DiffractionMultiTaskDataset(PROJECT_ROOT / args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = MultiTaskCNN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    total = 0
    image_correct = 0
    quality_abs_error = 0.0
    tp = 0.0
    fp = 0.0
    fn = 0.0
    primary_hit = 0
    primary_total = 0
    per_issue: Dict[str, Dict[str, float]] = {issue: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for issue in ISSUE_TYPES}

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch["image"])
            image_pred = torch.argmax(outputs["image_type_logits"], dim=1)
            image_correct += int((image_pred == batch["image_type"]).sum().item())
            total += int(batch["image"].shape[0])

            issue_probs = torch.sigmoid(outputs["issue_logits"])
            issue_pred = (issue_probs >= args.threshold).float()
            issue_true = batch["issues"].float()
            tp += float((issue_pred * issue_true).sum().item())
            fp += float((issue_pred * (1.0 - issue_true)).sum().item())
            fn += float(((1.0 - issue_pred) * issue_true).sum().item())

            for idx, issue in enumerate(ISSUE_TYPES):
                pred_col = issue_pred[:, idx]
                true_col = issue_true[:, idx]
                per_issue[issue]["tp"] += float((pred_col * true_col).sum().item())
                per_issue[issue]["fp"] += float((pred_col * (1.0 - true_col)).sum().item())
                per_issue[issue]["fn"] += float(((1.0 - pred_col) * true_col).sum().item())

            primary_pred = torch.argmax(issue_probs, dim=1)
            primary_mask = batch["primary_issue"] >= 0
            if primary_mask.any():
                primary_hit += int((primary_pred[primary_mask] == batch["primary_issue"][primary_mask]).sum().item())
                primary_total += int(primary_mask.sum().item())

            quality_abs_error += float(torch.abs(outputs["quality_score"] - batch["quality_score"]).sum().item())

    issue_micro_f1 = (2.0 * tp) / max(1e-8, 2.0 * tp + fp + fn)
    issue_macro_f1_values = []
    for values in per_issue.values():
        issue_macro_f1_values.append((2.0 * values["tp"]) / max(1e-8, 2.0 * values["tp"] + values["fp"] + values["fn"]))
    metrics = {
        "dataset_size": total,
        "image_type_accuracy": image_correct / max(1, total),
        "issue_micro_f1": issue_micro_f1,
        "issue_macro_f1": sum(issue_macro_f1_values) / max(1, len(issue_macro_f1_values)),
        "primary_issue_top1_accuracy": primary_hit / max(1, primary_total),
        "quality_mae": quality_abs_error / max(1, total),
        "threshold": args.threshold,
        "checkpoint": str(checkpoint_path),
    }

    for key, value in metrics.items():
        print(f"{key}={value}")

    output_path = PROJECT_ROOT / "runs" / "eval_small_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    print(f"metrics_path={output_path}")


if __name__ == "__main__":
    main()
