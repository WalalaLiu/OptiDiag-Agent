#!/usr/bin/env python3
"""Regenerate report-ready visualizations for existing demo cases."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from optidiag.utils.image_io import load_gray_image  # noqa: E402
from optidiag.utils.visualization import create_demo_visualization  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visual reports for demo cases.")
    parser.add_argument("--demo-dir", default="outputs/demo_cases", help="Directory made by make_demo_cases.py.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo_dir = PROJECT_ROOT / args.demo_dir
    summary_path = demo_dir / "summary.csv"
    visualizations_dir = demo_dir / "visualizations"
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    with summary_path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        image = load_gray_image(demo_dir / row["image_path"])
        with (demo_dir / row["result_path"]).open("r", encoding="utf-8") as file:
            payload = json.load(file)
        output_path = demo_dir / row["visualization_path"]
        create_demo_visualization(
            image,
            payload["analysis"],
            output_path,
            title=f"{row['case_id']}: {row['expected_issue']}",
            expected_issue=row["expected_issue"],
        )

    print(f"Regenerated {len(rows)} visualizations in {visualizations_dir}")


if __name__ == "__main__":
    main()

