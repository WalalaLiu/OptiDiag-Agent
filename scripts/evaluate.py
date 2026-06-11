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
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
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


def _issue_counts(issue_pred, issue_true):  # type: ignore[no-untyped-def]
    """Return micro precision/recall/F1 counts for multi-label predictions."""
    tp = float((issue_pred * issue_true).sum().item())
    fp = float((issue_pred * (1.0 - issue_true)).sum().item())
    fn = float(((1.0 - issue_pred) * issue_true).sum().item())
    precision = tp / max(1e-8, tp + fp)
    recall = tp / max(1e-8, tp + fn)
    f1 = (2.0 * precision * recall) / max(1e-8, precision + recall)
    return tp, fp, fn, precision, recall, f1


def _per_issue_metrics(pred_col, true_col):  # type: ignore[no-untyped-def]
    """Return precision/recall/F1 and positive counts for one issue."""
    issue_tp = float((pred_col * true_col).sum().item())
    issue_fp = float((pred_col * (1.0 - true_col)).sum().item())
    issue_fn = float(((1.0 - pred_col) * true_col).sum().item())
    precision = issue_tp / max(1e-8, issue_tp + issue_fp)
    recall = issue_tp / max(1e-8, issue_tp + issue_fn)
    f1 = (2.0 * precision * recall) / max(1e-8, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive_count": int(true_col.sum().item()),
        "predicted_positive_count": int(pred_col.sum().item()),
    }


def main() -> None:
    """Evaluate image type accuracy, issue F1, and quality MAE."""
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("Evaluation requires optional dependency: torch.") from exc

    from optidiag.constants import ISSUE_TYPES
    from optidiag.models.checkpoint import load_torch_checkpoint
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
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    total = 0
    image_correct = 0
    quality_abs_error = 0.0
    primary_hit = 0
    primary_total = 0
    issue_probs_all = []
    issue_true_all = []

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch["image"])
            image_pred = torch.argmax(outputs["image_type_logits"], dim=1)
            image_correct += int((image_pred == batch["image_type"]).sum().item())
            total += int(batch["image"].shape[0])

            issue_probs = torch.sigmoid(outputs["issue_logits"])
            issue_true = batch["issues"].float()
            issue_probs_all.append(issue_probs.cpu())
            issue_true_all.append(issue_true.cpu())

            primary_pred = torch.argmax(issue_probs, dim=1)
            primary_mask = batch["primary_issue"] >= 0
            if primary_mask.any():
                primary_hit += int((primary_pred[primary_mask] == batch["primary_issue"][primary_mask]).sum().item())
                primary_total += int(primary_mask.sum().item())

            quality_abs_error += float(torch.abs(outputs["quality_score"] - batch["quality_score"]).sum().item())

    issue_probs = torch.cat(issue_probs_all, dim=0)
    issue_true = torch.cat(issue_true_all, dim=0)
    thresholds = [float(value.strip()) for value in args.thresholds.split(",") if value.strip()]

    threshold_results = {}
    best_micro_threshold = thresholds[0]
    best_micro_f1 = -1.0
    best_macro_threshold = thresholds[0]
    best_macro_f1 = -1.0
    for threshold in thresholds:
        issue_pred = (issue_probs >= threshold).float()
        _, _, _, precision_micro, recall_micro, micro_f1 = _issue_counts(issue_pred, issue_true)
        macro_values = []
        for idx in range(len(ISSUE_TYPES)):
            pred_col = issue_pred[:, idx]
            true_col = issue_true[:, idx]
            macro_values.append(_per_issue_metrics(pred_col, true_col)["f1"])
        macro_f1 = sum(macro_values) / max(1, len(macro_values))
        predicted_positive_count = int(issue_pred.sum().item())
        threshold_results[str(threshold)] = {
            "issue_micro_f1": micro_f1,
            "issue_macro_f1": macro_f1,
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "predicted_positive_count": predicted_positive_count,
        }
        print(
            f"threshold={threshold:.1f} "
            f"issue_micro_f1={micro_f1:.4f} issue_macro_f1={macro_f1:.4f} "
            f"precision_micro={precision_micro:.4f} recall_micro={recall_micro:.4f} "
            f"predicted_positive_count={predicted_positive_count}"
        )
        if micro_f1 > best_micro_f1:
            best_micro_f1 = micro_f1
            best_micro_threshold = threshold
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_macro_threshold = threshold

    selected_pred = (issue_probs >= best_micro_threshold).float()
    per_issue: Dict[str, Dict[str, float]] = {}
    calibrated_thresholds: Dict[str, Dict[str, float]] = {}
    for idx, issue in enumerate(ISSUE_TYPES):
        true_col = issue_true[:, idx]
        pred_col = selected_pred[:, idx]
        per_issue[issue] = _per_issue_metrics(pred_col, true_col)

        best_issue_threshold = thresholds[0]
        best_issue_metrics = None
        for threshold in thresholds:
            threshold_pred_col = (issue_probs[:, idx] >= threshold).float()
            metrics_for_threshold = _per_issue_metrics(threshold_pred_col, true_col)
            if best_issue_metrics is None or metrics_for_threshold["f1"] > best_issue_metrics["f1"]:
                best_issue_threshold = threshold
                best_issue_metrics = metrics_for_threshold
        calibrated_thresholds[issue] = {
            "best_threshold": best_issue_threshold,
            **best_issue_metrics,
        }
        print(
            f"issue={issue} best_threshold={best_issue_threshold:.1f} "
            f"precision={best_issue_metrics['precision']:.4f} "
            f"recall={best_issue_metrics['recall']:.4f} "
            f"f1={best_issue_metrics['f1']:.4f} "
            f"true_positive_count={best_issue_metrics['true_positive_count']} "
            f"predicted_positive_count={best_issue_metrics['predicted_positive_count']}"
        )

    threshold_05 = threshold_results.get("0.5")
    if threshold_05 and threshold_05["predicted_positive_count"] == 0:
        print("warning=threshold_0.5_predicted_positive_count_is_zero; probabilities may be below the default threshold")

    metrics = {
        "dataset_size": total,
        "image_type_accuracy": image_correct / max(1, total),
        "issue_micro_f1": threshold_results[str(best_micro_threshold)]["issue_micro_f1"],
        "issue_macro_f1": threshold_results[str(best_macro_threshold)]["issue_macro_f1"],
        "threshold_sweep": threshold_results,
        "best_issue_threshold": best_micro_threshold,
        "best_micro_threshold": best_micro_threshold,
        "best_macro_threshold": best_macro_threshold,
        "per_class_thresholds": calibrated_thresholds,
        "per_issue": per_issue,
        "primary_issue_top1_accuracy": primary_hit / max(1, primary_total),
        "quality_mae": quality_abs_error / max(1, total),
        "threshold": args.threshold,
        "checkpoint": str(checkpoint_path),
    }

    for key, value in metrics.items():
        if key not in {"threshold_sweep", "per_issue"}:
            print(f"{key}={value}")

    output_path = PROJECT_ROOT / "runs" / "eval_small_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    print(f"metrics_path={output_path}")

    thresholds_path = PROJECT_ROOT / "runs" / "eval_thresholds.json"
    with thresholds_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "threshold_sweep": threshold_results,
                "best_micro_threshold": best_micro_threshold,
                "best_macro_threshold": best_macro_threshold,
                "per_class_thresholds": calibrated_thresholds,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"thresholds_path={thresholds_path}")


if __name__ == "__main__":
    main()
