"""Re-evaluate saved artifacts on the held-out test split and export reports.

Usage:
    python scripts/evaluate_models.py [--model-id MODEL_ID] [--export-csv]

Reloads fitted pipelines through the registry (the same code path the API
uses), recomputes metrics from real predictions and prints a comparison.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.ml.datasets import dedupe_dataset, load_dataset  # noqa: E402
from app.ml.evaluate import evaluate_pipeline  # noqa: E402
from app.ml.registry import ModelRegistry  # noqa: E402
from app.ml.train import split_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=None, help="Evaluate one model only")
    parser.add_argument("--export-csv", action="store_true", help="Write artifacts/evaluation_report.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    settings = get_settings()

    reg = ModelRegistry(settings.artifacts_path)
    try:
        meta = reg.reload()
    except Exception as exc:  # noqa: BLE001
        print(f"Registry unavailable: {exc}", file=sys.stderr)
        return 1

    data_file = settings.data_path / "SMSSpamCollection"
    if not data_file.exists():
        data_file = settings.data_path / "sms_spam.csv"
    df = load_dataset(data_file)
    df, _ = dedupe_dataset(df)
    _, test_df = split_dataset(df)  # identical seed => identical split

    ids = [args.model_id] if args.model_id else meta["available_model_ids"]
    rows = []
    for mid in ids:
        pipe = reg.get_pipeline(mid)
        metrics = evaluate_pipeline(pipe, test_df["text"], test_df["label"], include_auc=True)
        rows.append({"model_id": mid, **{
            k: metrics[k]
            for k in (
                "accuracy", "precision_spam", "recall_spam", "f1_spam",
                "macro_f1", "weighted_f1", "roc_auc_spam",
            )
            if k in metrics
        }})
        print(f"\n=== {mid} ===")
        print(json.dumps(
            {k: v for k, v in metrics.items() if k != "confusion_matrix"},
            indent=2,
        )[:900])
        print("Confusion matrix [rows=true, cols=pred] labels:", metrics["confusion_labels"])
        for row in metrics["confusion_matrix"]:
            print("  ", row)

    frame = pd.DataFrame(rows)
    print("\nComparison (test split):")
    print(frame.to_string(index=False))

    if args.export_csv:
        out = settings.artifacts_path / "evaluation_report.csv"
        frame.to_csv(out, index=False)
        print(f"Exported -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
