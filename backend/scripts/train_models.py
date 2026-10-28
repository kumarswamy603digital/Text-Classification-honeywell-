"""Train, tune, compare and select all four traditional ML models.

Usage:
    python scripts/train_models.py [--use-char-ngrams] [--quick]

Protocol
--------
1. Load + dedupe dataset, stratified 80/20 train/test split (seed=42).
2. Choose TF-IDF variant on TRAIN data via CV (char n-grams only if they
   measurably improve macro-F1).
3. GridSearchCV (3-fold stratified, macro-F1) per classifier on TRAIN data.
4. Select model with best VALIDATION macro-F1.
5. Evaluate every model once on the untouched TEST split for the report.
6. Save pipelines (.joblib), metadata (.json) and metrics (.csv) to artifacts/.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.ml.datasets import compute_stats, dedupe_dataset, load_dataset  # noqa: E402
from app.ml.evaluate import evaluate_pipeline  # noqa: E402
from app.ml.registry import ModelRegistry  # noqa: E402
from app.ml.train import (  # noqa: E402
    MODEL_VERSION,
    build_metadata,
    classifier_specs,
    save_pipeline,
    split_dataset,
    tune_model,
    tune_tfidf_variant,
    write_metadata,
)

logger = logging.getLogger("train_models")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--use-char-ngrams",
        action="store_true",
        help="Include a char_wb TF-IDF variant in the feature experiment",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduced search space for fast CI runs (documented in README)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    settings = get_settings()
    t0 = time.time()

    data_file = settings.data_path / "SMSSpamCollection"
    if not data_file.exists():
        data_file = settings.data_path / "sms_spam.csv"
    if not data_file.exists():
        print("Dataset missing. Run: python scripts/download_data.py", file=sys.stderr)
        return 1

    df = load_dataset(data_file)
    df, n_dup = dedupe_dataset(df)
    stats = compute_stats(df, n_duplicates=n_dup)

    train_df, test_df = split_dataset(df)
    split_info = {
        "strategy": "stratified 80/20 train/test, seed=%d" % settings.random_seed,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "duplicates_removed_before_split": n_dup,
    }
    logger.info("Split: train=%d test=%d", len(train_df), len(test_df))

    # ---- Step 1: TF-IDF variant comparison (train data only) ----
    cv_folds = 2 if args.quick else 3
    tfidf_params, tfidf_score = tune_tfidf_variant(
        train_df, use_char_ngrams=args.use_char_ngrams, cv_folds=cv_folds
    )
    logger.info("Chosen TF-IDF config %s (proxy macro-F1 %.4f)", tfidf_params, tfidf_score)

    # ---- Step 2: tune each classifier on TRAIN with CV ----
    specs = classifier_specs()
    tuned: dict[str, tuple] = {}
    for mid, spec in specs.items():
        logger.info("Tuning %s ...", spec.name)
        gs = tune_model(spec, train_df, tfidf_params, cv_folds=cv_folds)
        tuned[mid] = (spec, gs)
        logger.info(
            "  %s best CV macro-F1 = %.4f with %s",
            spec.name,
            gs.best_score_,
            gs.best_params_,
        )

    # ---- Step 3: select best on VALIDATION (CV) macro-F1 ----
    selected_id = max(tuned, key=lambda k: tuned[k][1].best_score_)
    logger.info("SELECTED model by validation macro-F1: %s", selected_id)

    # ---- Step 4: final evaluation on untouched TEST split ----
    artifacts = settings.artifacts_path
    results = []
    for mid, (spec, gs) in tuned.items():
        pipe = gs.best_estimator_
        metrics = evaluate_pipeline(
            pipe, test_df["text"], test_df["label"],
            include_auc=spec.supports_predict_proba,
        )
        save_pipeline(pipe, artifacts / f"{mid}.joblib")
        results.append(
            {
                "model_id": mid,
                "name": spec.name,
                "version": MODEL_VERSION,
                "supports_predict_proba": spec.supports_predict_proba,
                "notes": spec.notes,
                "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
                "validation_macro_f1": float(gs.best_score_),
                "test_metrics": metrics,
            }
        )
        logger.info(
            "TEST %s: acc=%.4f spamF1=%.4f macroF1=%.4f",
            mid,
            metrics["accuracy"],
            metrics["f1_spam"],
            metrics["macro_f1"],
        )

    # ---- Step 5: persist registry metadata + CSV export ----
    metadata = build_metadata(results, selected_id, tfidf_params, stats, split_info)
    write_metadata(metadata, artifacts / "model_metadata.json")

    csv_rows = []
    for r in results:
        tm = r["test_metrics"]
        csv_rows.append(
            {
                "model_id": r["model_id"],
                "name": r["name"],
                "validation_macro_f1": round(r["validation_macro_f1"], 4),
                "test_accuracy": round(tm["accuracy"], 4),
                "test_precision_spam": round(tm["precision_spam"], 4),
                "test_recall_spam": round(tm["recall_spam"], 4),
                "test_f1_spam": round(tm["f1_spam"], 4),
                "test_macro_f1": round(tm["macro_f1"], 4),
                "test_weighted_f1": round(tm["weighted_f1"], 4),
                "test_roc_auc_spam": tm.get("roc_auc_spam"),
                "selected": r["model_id"] == selected_id,
            }
        )
    pd.DataFrame(csv_rows).to_csv(artifacts / "model_metrics.csv", index=False)

    # Verify the registry can reload what we just wrote
    reg = ModelRegistry(artifacts)
    reg.reload()
    assert reg.selected_model_id == selected_id
    reg.get_pipeline(selected_id)

    logger.info("Done in %.1fs. Artifacts in %s", time.time() - t0, artifacts)
    print("\nSummary (test split):")
    print(pd.DataFrame(csv_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
