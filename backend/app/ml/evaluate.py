"""Evaluation metrics computed from REAL fitted-model predictions.

Nothing here is hardcoded: every number originates from sklearn metric
functions applied to a fitted pipeline and a held-out split.
"""
from __future__ import annotations

import logging
import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.ml.datasets import CLASS_NAMES

logger = logging.getLogger(__name__)


def evaluate_pipeline(pipe, X_test, y_test, include_auc: bool = False) -> dict:
    """Full metric bundle for one fitted pipeline on a held-out split."""
    y_true = np.asarray(y_test)
    y_pred = pipe.predict(X_test)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )

    per_class = {
        name: {
            "precision": float(report[name]["precision"]),
            "recall": float(report[name]["recall"]),
            "f1": float(report[name]["f1-score"]),
            "support": int(report[name]["support"]),
        }
        for name in CLASS_NAMES
    }

    out: dict = {
        "n_evaluated": int(len(y_true)),
        "accuracy": acc,
        "precision_spam": prec,
        "recall_spam": rec,
        "f1_spam": f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm.tolist(),
        "confusion_labels": CLASS_NAMES,
        "per_class": per_class,
    }

    if include_auc and hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_true, proba))
            pr_auc = float(average_precision_score(y_true, proba))
            out["roc_auc_spam"] = None if math.isnan(auc) else auc
            out["pr_auc_spam"] = None if math.isnan(pr_auc) else pr_auc
            # threshold sweep so the UI can discuss error costs
            prec_t, rec_t, thr = precision_recall_curve(y_true, proba)
            if len(thr):
                idx = int(np.argmax(np.minimum(rec_t[:-1], prec_t[:-1])))
                out["balanced_threshold"] = {
                    "threshold": float(thr[idx]),
                    "precision": float(prec_t[idx]),
                    "recall": float(rec_t[idx]),
                }
        except Exception as exc:  # noqa: BLE001 - AUC is optional
            logger.warning("AUC computation skipped: %s", exc)

    tn, fp, fn, tp = cm.ravel()
    out["error_breakdown"] = {
        "true_negatives": int(tn),
        "false_positives_spam_flag_on_ham": int(fp),
        "false_negives_spam_missed": int(fn),
        "true_positives": int(tp),
    }
    return out
