#!/usr/bin/env python3
"""Run the OptiDiag FastAPI server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OptiDiag API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Missing dependency: install requirements.txt before running the API.") from exc

    args = parse_args()
    uvicorn.run("optidiag.api.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

