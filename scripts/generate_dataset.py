#!/usr/bin/env python3
"""Generate a small synthetic diffraction diagnosis dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from optidiag.simulation.dataset_generator import DatasetGenerator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic diffraction MVP data.")
    parser.add_argument("--num", type=int, default=200, help="Number of images to generate.")
    parser.add_argument("--out", type=str, default="data/simulated/diffraction_mvp", help="Output dataset directory.")
    parser.add_argument("--image-size", type=int, default=256, help="Output image size in pixels.")
    parser.add_argument("--grid-size", type=int, default=512, help="FFT simulation grid size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = DatasetGenerator(image_size=args.image_size, grid_size=args.grid_size, seed=args.seed)
    labels_path = generator.generate(out_dir=PROJECT_ROOT / args.out, num_images=args.num)
    print(f"Generated {args.num} images")
    print(f"Labels: {labels_path}")


if __name__ == "__main__":
    main()

