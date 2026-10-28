"""Model definitions and reproducible training routines.

Every model is a full sklearn Pipeline:
    TextPreprocessor -> TfidfVectorizer -> Classifier
The pipeline is saved as ONE artifact, so inference cannot drift from training
preprocessing and the vectorizer vocabulary is always the fitted one.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

from app.core.config import get_settings
from app.ml.features import build_tfidf
from app.ml.preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)

MODEL_VERSION = "1.0.0"

@dataclass
class ModelSpec:
    model_id: str
    name: str
    supports_predict_proba: bool
    estimator_cls: type
    param_grid: dict = field(default_factory=dict)
    notes: str = ""
    estimator_kwargs: dict = field(default_factory=dict)

    def new_estimator(self):
        """Fresh estimator instance (pipelines must not share mutable state)."""
        return self.estimator_cls(**self.estimator_kwargs)


def classifier_specs() -> dict[str, ModelSpec]:
    """The four traditional ML classifiers required by the project brief."""
    return {
        "mnb": ModelSpec(
            model_id="mnb",
            name="Multinomial Naive Bayes",
            supports_predict_proba=True,
            estimator_cls=MultinomialNB,
            param_grid={"clf__alpha": [0.01, 0.05, 0.1, 0.3, 1.0]},
            notes="Fast probabilistic baseline; assumes feature independence.",
        ),
        "logreg": ModelSpec(
            model_id="logreg",
            name="Logistic Regression",
            supports_predict_proba=True,
            estimator_cls=LogisticRegression,
            estimator_kwargs={"max_iter": 2000},
            param_grid={"clf__C": [0.1, 0.5, 1.0, 4.0, 10.0]},
            notes="Linear probabilistic model; probabilities are well calibrated "
                  "in practice for text tasks.",
        ),
        "linearsvc": ModelSpec(
            model_id="linearsvc",
            name="Linear SVM",
            supports_predict_proba=False,
            estimator_cls=LinearSVC,
            estimator_kwargs={"max_iter": 2000},
            param_grid={"clf__C": [0.1, 1.0, 4.0]},
            notes="Margin-based classifier. Decision scores are NOT probabilities; "
                  "the API reports no confidence for this model.",
        ),
        "cnb": ModelSpec(
            model_id="cnb",
            name="Complement Naive Bayes",
            supports_predict_proba=True,
            estimator_cls=ComplementNB,
            param_grid={"clf__alpha": [0.01, 0.05, 0.1, 0.3, 1.0]},
            notes="Designed for imbalanced text data; uses complements of class stats.",
        ),
    }


def build_pipeline(spec: ModelSpec, tfidf_params: dict) -> object:
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("clean", TextPreprocessor()),
            ("tfidf", build_tfidf(**tfidf_params)),
            ("clf", spec.new_estimator()),
        ]
    )


def split_dataset(df: pd.DataFrame, seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 80/20 train/test split. The test set stays untouched until
    the single final evaluation of the selected model."""
    s = get_settings()
    return train_test_split(
        df,
        test_size=0.2,
        random_state=seed if seed is not None else s.random_seed,
        stratify=df["label"],
        shuffle=True,
    )


def tune_tfidf_variant(train_df: pd.DataFrame, use_char_ngrams: bool, cv_folds: int, n_jobs: int = -1) -> tuple[dict, float]:
    """Compare word-only vs word+char TF-IDF using a cheap proxy (MNB) under
    the same CV protocol, so char features are kept ONLY if they measurably
    help. Returns (tfidf_params, macro-F1)."""
    variants = [
        {"ngram_range": (1, 1), "min_df": 2, "max_df": 0.95, "max_features": 50000,
         "use_char_ngrams": False},
        {"ngram_range": (1, 2), "min_df": 2, "max_df": 0.95, "max_features": 50000,
         "use_char_ngrams": False},
    ]
    if use_char_ngrams:
        variants.append(
            {"ngram_range": (1, 2), "min_df": 2, "max_df": 0.95, "max_features": 80000,
             "use_char_ngrams": True}
        )

    spec = ModelSpec("proxy", "MNB proxy", True, MultinomialNB)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=get_settings().random_seed)
    best_params, best_score = None, -1.0
    for params in variants:
        pipe = build_pipeline(spec, params)
        gs = GridSearchCV(
            pipe, param_grid={"clf__alpha": [0.1]}, cv=cv,
            scoring="f1_macro", n_jobs=n_jobs, refit=False,
        )
        gs.fit(train_df["text"], train_df["label"])
        score = float(gs.best_score_)
        logger.info("TF-IDF variant %s -> macro-F1 %.4f", params, score)
        if score > best_score:
            best_params, best_score = params, score
    assert best_params is not None
    return best_params, best_score


def tune_model(
    spec: ModelSpec,
    train_df: pd.DataFrame,
    tfidf_params: dict,
    cv_folds: int = 3,
    n_jobs: int = -1,
) -> GridSearchCV:
    """Grid-search the classifier hyperparameters (with TF-IDF params folded
    into the CV) on TRAIN data only, scored by macro-F1."""
    pipe = build_pipeline(spec, tfidf_params)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=get_settings().random_seed)
    gs = GridSearchCV(
        pipe,
        param_grid=spec.param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=n_jobs,
        refit=True,
        return_train_score=False,
    )
    gs.fit(train_df["text"], train_df["label"])
    return gs


def save_pipeline(pipe, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    logger.info("Saved fitted pipeline -> %s", path)


def load_pipeline(path: Path):
    from app.ml.registry import ModelRegistryError

    if not path.exists():
        raise ModelRegistryError(f"Model artifact not found: {path.name}")
    return joblib.load(path)


def build_metadata(
    results: list[dict],
    selected_id: str,
    tfidf_params: dict,
    dataset_stats: dict,
    split_info: dict,
) -> dict:
    return {
        "task": "sms_spam_detection",
        "positive_class": "spam",
        "version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "selection_criterion": "best validation macro-F1 (3-fold stratified CV on train split)",
        "final_evaluation": "held-out 20% test split, evaluated once after selection",
        "tfidf_params": {k: list(v) if isinstance(v, tuple) else v for k, v in tfidf_params.items()},
        "dataset": {
            "name": dataset_stats.get("source", {}).get("name"),
            "n_samples": dataset_stats.get("n_samples"),
            "class_counts": dataset_stats.get("class_counts"),
        },
        "split": split_info,
        "models": results,
        "selected_model_id": selected_id,
        "available_model_ids": [r["model_id"] for r in results],
    }


def _sanitize(obj):
    """Recursively replace NaN/Inf floats with None so the JSON is
    browser-parseable (JS JSON.parse rejects NaN literals)."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def write_metadata(metadata: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = _sanitize(metadata)
    path.write_text(json.dumps(clean, indent=2, default=str), encoding="utf-8")
    logger.info("Model metadata saved -> %s", path)
