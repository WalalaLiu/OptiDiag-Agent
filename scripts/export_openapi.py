#!/usr/bin/env python3
"""Export the FastAPI OpenAPI schema to a JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from optidiag.api.main import app  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export OpenAPI schema.")
    parser.add_argument("--out", default="docs/openapi.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = PROJECT_ROOT / args.out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(app.openapi(), file, ensure_ascii=False, indent=2)
    print(f"OpenAPI schema written to {output_path}")


if __name__ == "__main__":
    main()

