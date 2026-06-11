#!/usr/bin/env python3
"""Check class and issue balance for an OptiDiag labels.csv file."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Counter as CounterType, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from optidiag.constants import IMAGE_TYPES, ISSUE_TYPES  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Check labels.csv class and issue balance.")
    parser.add_argument("--labels", required=True, help="Path to labels.csv")
    parser.add_argument("--imbalance-ratio", type=float, default=3.0, help="Warn when max/min count exceeds this ratio.")
    return parser.parse_args()


def _ratio(counter: CounterType[str], labels: Iterable[str]) -> float:
    """Return max/min ratio for labels with zero-aware smoothing."""
    counts = [counter.get(label, 0) for label in labels]
    if not counts:
        return 0.0
    return max(counts) / max(1, min(counts))


def _print_counter(title: str, counter: CounterType[str], labels: Iterable[str]) -> None:
    """Print a stable counter table."""
    print(title)
    for label in labels:
        print(f"  {label}: {counter.get(label, 0)}")


def _load_issues(row: Dict[str, str]) -> List[str]:
    """Load issues from JSON list or issue_scores fallback."""
    try:
        issues = json.loads(row.get("issues", "[]"))
        if isinstance(issues, list):
            return [str(issue) for issue in issues]
    except json.JSONDecodeError:
        pass

    try:
        scores = json.loads(row.get("issue_scores", "{}"))
        return [issue for issue, score in scores.items() if float(score) >= 0.2]
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def main() -> None:
    """Print class balance and imbalance warnings."""
    args = parse_args()
    labels_path = Path(args.labels)
    if not labels_path.is_absolute():
        labels_path = PROJECT_ROOT / labels_path

    with labels_path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit(f"No rows found: {labels_path}")

    image_counter: CounterType[str] = Counter()
    issue_counter: CounterType[str] = Counter()
    primary_counter: CounterType[str] = Counter()
    severity_counter: CounterType[str] = Counter()

    for row in rows:
        image_counter[row["image_type"]] += 1
        primary_counter[row["primary_issue"]] += 1
        severity_counter[row["severity"]] += 1
        for issue in _load_issues(row):
            issue_counter[issue] += 1

    print(f"labels={labels_path}")
    print(f"total_rows={len(rows)}")
    _print_counter("image_type_counts", image_counter, IMAGE_TYPES)
    _print_counter("issue_counts", issue_counter, ISSUE_TYPES)
    _print_counter("primary_issue_counts", primary_counter, ISSUE_TYPES)
    _print_counter("severity_counts", severity_counter, ["none", "low", "medium", "high"])

    image_ratio = _ratio(image_counter, IMAGE_TYPES)
    issue_ratio = _ratio(issue_counter, ISSUE_TYPES)
    primary_ratio = _ratio(primary_counter, ISSUE_TYPES)
    print(f"image_type_imbalance_ratio={image_ratio:.2f}")
    print(f"issue_imbalance_ratio={issue_ratio:.2f}")
    print(f"primary_issue_imbalance_ratio={primary_ratio:.2f}")

    warnings = []
    if image_ratio > args.imbalance_ratio:
        warnings.append("image_type distribution is seriously imbalanced")
    if issue_ratio > args.imbalance_ratio:
        warnings.append("issue distribution is seriously imbalanced")
    if primary_ratio > args.imbalance_ratio:
        warnings.append("primary_issue distribution is seriously imbalanced")
    missing_images = [label for label in IMAGE_TYPES if image_counter.get(label, 0) == 0]
    missing_issues = [label for label in ISSUE_TYPES if primary_counter.get(label, 0) == 0]
    if missing_images:
        warnings.append(f"missing image types: {', '.join(missing_images)}")
    if missing_issues:
        warnings.append(f"missing primary issues: {', '.join(missing_issues)}")

    if warnings:
        print("balance_status=warning")
        for warning in warnings:
            print(f"warning={warning}")
    else:
        print("balance_status=ok")


if __name__ == "__main__":
    main()

