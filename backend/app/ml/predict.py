"""Prediction service: turns texts into labels, probabilities and
feature-based explanations using REAL fitted pipelines from the registry."""
from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone

import numpy as np

from app.core.config import get_settings
from app.ml.datasets import CLASS_NAMES
from app.ml.registry import ModelRegistry, ModelRegistryError

logger = logging.getLogger(__name__)

# Matches the default TfidfVectorizer token pattern (word characters incl. unicode)
_TOKEN_RE = re.compile(r"\w{1,}")


class EmptyTextInputError(ValueError):
    """Raised when a prediction request contains no usable text."""


def _validate_texts(texts: list[str]) -> list[str]:
    settings = get_settings()
    cleaned: list[str] = []
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            raise EmptyTextInputError(f"Item {i}: input must be a string")
        stripped = t.strip()
        if not stripped:
            raise EmptyTextInputError(f"Item {i}: input text is empty")
        if len(t) > settings.max_text_length:
            raise EmptyTextInputError(
                f"Item {i}: input exceeds {settings.max_text_length} characters"
            )
        cleaned.append(t)
    return cleaned


def _top_terms_for_text(clean_text: str, vocab: dict, max_terms: int = 12) -> list[str]:
    """Tokens of the (already pipeline-cleaned) text that exist in the fitted
    vocabulary. Uses the same 1-2 gram expansion as the vectorizer so linear
    model lookup hits bigram weights too."""
    words = _TOKEN_RE.findall(clean_text.lower())
    unigrams = [w for w in words if w in vocab]
    bigrams = [f"{a} {b}" for a, b in zip(words, words[1:]) if f"{a} {b}" in vocab]
    return (unigrams + bigrams)[:max_terms]


def _linear_coefficients(pipe, clf_name: str) -> np.ndarray | None:
    """Predicted-class coefficient row for linear models (LogReg / LinearSVC)."""
    clf = pipe.named_steps.get("clf")
    if clf_name.endswith("NB"):
        return None
    if hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_)
        return coef
    return None


def _nb_log_ratio(pipe) -> np.ndarray | None:
    """Feature log-probability ratio log P(f|spam) - log P(f|ham) for NB models."""
    clf = pipe.named_steps.get("clf")
    if hasattr(clf, "feature_log_prob_"):
        logp = np.asarray(clf.feature_log_prob_)  # (n_classes, n_features)
        if logp.shape[0] == 2:
            return logp[1] - logp[0]
    return None


def explain_prediction(pipe, clean_text: str, label_idx: int, model_name: str) -> list[dict]:
    """Instance-specific, weight-based explanation.

    IMPORTANT: these are the highest-weight vocabulary terms *present in this
    text* — informative feature attributions for a linear model, not a proof
    of causality. For NB models we use the class log-probability ratio.
    """
    tfidf = pipe.named_steps["tfidf"]
    vocab = tfidf.vocabulary_
    terms = _top_terms_for_text(clean_text, vocab)
    if not terms:
        return []

    weights: np.ndarray | None = None
    clf_name = type(pipe.named_steps["clf"]).__name__
    if clf_name.endswith("NB"):
        # Naive Bayes: use per-feature class log-probability ratio
        weights = _nb_log_ratio(pipe)
    if weights is None:
        coef = _linear_coefficients(pipe, clf_name)
        if coef is not None:
            weights = coef[min(label_idx, coef.shape[0] - 1)]
    if weights is None:
        return []

    scored = []
    for term in terms:
        idx = vocab.get(term)
        if idx is None or idx >= len(weights):
            continue
        scored.append({"term": term, "weight": float(weights[idx])})
    scored.sort(key=lambda d: abs(d["weight"]), reverse=True)
    return scored[:8]


class PredictionService:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def predict(self, texts: list[str], model_id: str | None = None) -> list[dict]:
        """Classify one or more texts with a fitted pipeline from the registry."""
        texts = _validate_texts(texts)
        mid = model_id or self.registry.selected_model_id
        info = self.registry.get_model_info(mid)
        pipe = self.registry.get_pipeline(mid)

        X = np.asarray(texts, dtype=object)
        t0 = datetime.now(timezone.utc)
        preds = pipe.predict(X)
        latency_ms = (datetime.now(timezone.utc) - t0).total_seconds() * 1000.0 / max(1, len(texts))

        # cleaned text per item, for explanations (reuse pipeline's preprocessor)
        cleaner = pipe.named_steps["clean"]

        probas = None
        if info["supports_predict_proba"] and hasattr(pipe, "predict_proba"):
            probas = pipe.predict_proba(X)

        results: list[dict] = []
        for i, (raw, p) in enumerate(zip(texts, preds)):
            label_idx = int(p)
            res = {
                "prediction_id": str(uuid.uuid4()),
                "model_id": mid,
                "model_name": info["name"],
                "model_version": info.get("version", "1.0.0"),
                "predicted_label": CLASS_NAMES[label_idx],
                "label_display": "Spam" if label_idx == 1 else "Legitimate (ham)",
                "label_index": label_idx,
                "status": "success",
                "latency_ms": round(latency_ms, 3),
                "timestamp": t0.isoformat(),
            }
            clean = cleaner.clean_one(raw)
            res["explanation_terms"] = explain_prediction(pipe, clean, label_idx, mid)

            if probas is not None:
                res["probabilities"] = {
                    CLASS_NAMES[j]: float(probas[i][j]) for j in range(len(CLASS_NAMES))
                }
                res["confidence"] = float(probas[i][label_idx])
                res["confidence_type"] = "estimated_class_probability"
            else:
                res["probabilities"] = None
                res["confidence"] = None
                res["confidence_type"] = None
                res["confidence_note"] = (
                    f"{info['name']} does not expose calibrated class probabilities; "
                    "no confidence figure is reported rather than a converted score."
                )
            results.append(res)
        return results
