"""Acquire the UCI SMS Spam Collection dataset.

Usage:
    python scripts/download_data.py [--force]
Exits non-zero with a clear message on failure (never fabricates data).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.ml.datasets import DatasetError, download_raw, load_dataset, validate_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    settings = get_settings()
    data_dir = Path(args.data_dir) if args.data_dir else settings.data_path

    try:
        raw = download_raw(data_dir, force=args.force)
    except DatasetError as exc:
        print(f"DOWNLOAD FAILED: {exc}", file=sys.stderr)
        return 1

    # If a pre-cleaned CSV was supplied instead, honour it
    csv_alt = data_dir / "sms_spam.csv"
    source = csv_alt if csv_alt.exists() else raw
    try:
        df = load_dataset(source)
    except DatasetError as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 2

    print(f"Dataset OK: {len(df)} rows from {source}")
    print(df["label"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
