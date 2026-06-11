#!/usr/bin/env python3
"""Inspect issue logits and probabilities from a trained checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Inspect model issue predictions.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--show-samples", type=int, default=10)
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
    """Print logits/probability diagnostics and sample predictions."""
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("Prediction inspection requires optional dependency: torch.") from exc

    from optidiag.constants import IMAGE_TYPES, ISSUE_TYPES
    from optidiag.models.checkpoint import load_torch_checkpoint
    from optidiag.models.dataset import DiffractionMultiTaskDataset
    from optidiag.models.network import MultiTaskCNN

    args = parse_args()
    device = _prepare_device(args.device)
    dataset = DiffractionMultiTaskDataset(PROJECT_ROOT / args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = MultiTaskCNN().to(device)
    checkpoint = load_torch_checkpoint(PROJECT_ROOT / args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    logits_all = []
    probs_all = []
    true_all = []
    image_pred_all = []
    primary_pred_all = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch["image"])
            logits = outputs["issue_logits"]
            probs = torch.sigmoid(logits)
            logits_all.append(logits.cpu())
            probs_all.append(probs.cpu())
            true_all.append(batch["issues"].cpu())
            image_pred_all.append(torch.argmax(outputs["image_type_logits"], dim=1).cpu())
            primary_pred_all.append(torch.argmax(probs, dim=1).cpu())

    logits_all = torch.cat(logits_all, dim=0)
    probs_all = torch.cat(probs_all, dim=0)
    true_all = torch.cat(true_all, dim=0)
    image_pred_all = torch.cat(image_pred_all, dim=0)
    primary_pred_all = torch.cat(primary_pred_all, dim=0)

    print(
        f"issue_logits min={float(logits_all.min()):.4f} max={float(logits_all.max()):.4f} "
        f"mean={float(logits_all.mean()):.4f}"
    )
    print(
        f"issue_sigmoid_probability min={float(probs_all.min()):.4f} max={float(probs_all.max()):.4f} "
        f"mean={float(probs_all.mean()):.4f}"
    )

    true_counts = true_all.sum(dim=0)
    print("true_positive_counts")
    for issue, count in zip(ISSUE_TYPES, true_counts.tolist()):
        print(f"  {issue}: {int(count)}")

    for threshold in [0.1, 0.2, 0.3, 0.5]:
        pred_counts = (probs_all >= threshold).float().sum(dim=0)
        print(f"predicted_positive_counts threshold={threshold:.1f}")
        for issue, count in zip(ISSUE_TYPES, pred_counts.tolist()):
            print(f"  {issue}: {int(count)}")

    print("sample_predictions")
    rows_to_show = min(args.show_samples, len(dataset))
    for idx in range(rows_to_show):
        row = dataset.rows[idx]
        probs = probs_all[idx]
        top_probs, top_indices = torch.topk(probs, k=3)
        top3 = [
            f"{ISSUE_TYPES[int(issue_idx)]}:{float(prob):.3f}"
            for prob, issue_idx in zip(top_probs.tolist(), top_indices.tolist())
        ]
        print(
            f"sample={idx} "
            f"image_type_true={row['image_type']} image_type_pred={IMAGE_TYPES[int(image_pred_all[idx])]} "
            f"primary_issue_true={row['primary_issue']} primary_issue_pred={ISSUE_TYPES[int(primary_pred_all[idx])]} "
            f"issue_probability_top3={top3}"
        )


if __name__ == "__main__":
    main()

