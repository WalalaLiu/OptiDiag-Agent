#!/usr/bin/env python3
"""Evaluate rule-based smoke diagnostics or an optional checkpoint."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from optidiag.reasoning.rules import analyze_image_rule_based  # noqa: E402
from optidiag.utils.image_io import load_gray_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MVP rule-based diagnosis on labels.csv.")
    parser.add_argument("--data", default="data/simulated/diffraction_mvp")
    parser.add_argument("--limit", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = PROJECT_ROOT / args.data
    labels_path = dataset_dir / "labels.csv"
    rows = []
    with labels_path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    hit = 0
    total = 0
    for row in rows[: args.limit]:
        image = load_gray_image(dataset_dir / row["image_path"])
        result = analyze_image_rule_based(image)
        detected = {issue["type"] for issue in result["issues"]}
        hit += int(row["primary_issue"] in detected)
        total += 1
    print(f"primary_issue_rule_hit_rate={hit / max(1, total):.3f} ({hit}/{total})")


if __name__ == "__main__":
    main()

