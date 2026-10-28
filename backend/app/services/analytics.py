"""Analytics service: exposes REAL training/evaluation results to the UI.

All data comes from artifacts written by scripts/train_models.py. If a file
is missing, the service returns an honest "unavailable" state instead of
inventing numbers.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class AnalyticsService:
    def __init__(self) -> None:
        self.artifacts = get_settings().artifacts_path

    def _read_json(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse %s: %s", path.name, exc)
            return None

    @property
    def metadata(self) -> dict | None:
        return self._read_json(self.artifacts / "model_metadata.json")

    @property
    def dataset_stats(self) -> dict | None:
        return self._read_json(self.artifacts / "dataset_stats.json")

    @property
    def eda_extra(self) -> dict | None:
        return self._read_json(self.artifacts / "eda_extra.json")

    def overview(self) -> dict | None:
        meta = self.metadata
        if meta is None:
            return None
        selected_id = meta.get("selected_model_id")
        selected = next(
            (m for m in meta.get("models", []) if m["model_id"] == selected_id), None
        )
        return {
            "task": meta.get("task"),
            "dataset": meta.get("dataset", {}),
            "dataset_stats": self.dataset_stats,
            "selected_model": selected,
            "selected_metrics": (selected or {}).get("test_metrics"),
            "n_models_trained": len(meta.get("models", [])),
            "trained_at": meta.get("trained_at"),
            "selection_criterion": meta.get("selection_criterion"),
        }

    def model_comparison(self) -> dict | None:
        meta = self.metadata
        if meta is None:
            return None
        rows = []
        for m in meta.get("models", []):
            tm = m.get("test_metrics") or {}
            rows.append(
                {
                    "model_id": m["model_id"],
                    "name": m["name"],
                    "accuracy": tm.get("accuracy"),
                    "precision_spam": tm.get("precision_spam"),
                    "recall_spam": tm.get("recall_spam"),
                    "f1_spam": tm.get("f1_spam"),
                    "macro_f1": tm.get("macro_f1"),
                    "weighted_f1": tm.get("weighted_f1"),
                    "roc_auc_spam": tm.get("roc_auc_spam"),
                    "validation_macro_f1": m.get("validation_macro_f1"),
                    "best_params": m.get("best_params", {}),
                    "selected": m["model_id"] == meta.get("selected_model_id"),
                }
            )
        n_evaluated = next(
            (
                (m.get("test_metrics") or {}).get("n_evaluated")
                for m in meta.get("models", [])
                if (m.get("test_metrics") or {}).get("n_evaluated")
            ),
            None,
        )
        return {
            "protocol": {
                "split": meta.get("split"),
                "selection_criterion": meta.get("selection_criterion"),
                "final_evaluation": meta.get("final_evaluation"),
                "positive_class": meta.get("positive_class", "spam"),
                "n_evaluated": n_evaluated,
            },
            "rows": rows,
        }

    def confusion_matrix(self, model_id: str | None = None) -> dict | None:
        meta = self.metadata
        if meta is None:
            return None
        mid = model_id or meta.get("selected_model_id")
        model = next((m for m in meta.get("models", []) if m["model_id"] == mid), None)
        if model is None:
            return None
        tm = model.get("test_metrics") or {}
        if "confusion_matrix" not in tm:
            return None
        return {
            "model_id": mid,
            "labels": tm.get("confusion_labels", ["ham", "spam"]),
            "matrix": tm["confusion_matrix"],
            "n_evaluated": tm.get("n_evaluated"),
            "positive_class": meta.get("positive_class", "spam"),
            "error_breakdown": tm.get("error_breakdown", {}),
        }
