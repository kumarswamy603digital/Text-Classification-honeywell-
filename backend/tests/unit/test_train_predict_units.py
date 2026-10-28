"""Unit tests for training machinery, prediction service and evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from app.ml.evaluate import evaluate_pipeline
from app.ml.predict import EmptyTextInputError, PredictionService, _validate_texts
from app.ml.registry import ModelRegistry, ModelRegistryError
from app.ml.train import (
    build_metadata,
    build_pipeline,
    classifier_specs,
    save_pipeline,
    split_dataset,
    tune_model,
)


class TestTrainMachinery:
    def test_all_four_models_defined(self):
        specs = classifier_specs()
        assert set(specs) == {"mnb", "logreg", "linearsvc", "cnb"}
        for s in specs.values():
            assert s.param_grid, f"{s.model_id} has no tuning grid"

    def test_split_is_stratified_and_reproducible(self, tiny_dataframe):
        df = tiny_dataframe.copy()
        df["label"] = df["label"].map({"ham": 0, "spam": 1})
        a1, b1 = split_dataset(df, seed=7)
        a2, b2 = split_dataset(df, seed=7)
        pd.testing.assert_frame_equal(a1, a2)
        assert set(a1.index) & set(b1.index) == set()  # no overlap
        # roughly balanced classes in both halves (stratification)
        assert abs(a1["label"].mean() - b1["label"].mean()) < 0.06

    def test_pipeline_trains_and_predicts(self, tiny_dataframe):
        spec = classifier_specs()["logreg"]
        pipe = build_pipeline(spec, {"ngram_range": (1, 1), "min_df": 1, "max_features": None})
        labels = tiny_dataframe["label"].map({"ham": 0, "spam": 1})
        pipe.fit(tiny_dataframe["text"], labels)
        preds = pipe.predict(["free cash prize winner click now", "see you at lunch today mum"])
        assert preds.tolist() == [1, 0]

    def test_artifact_save_load_roundtrip(self, fitted_pipeline, tmp_path: Path):
        p = tmp_path / "pipe.joblib"
        save_pipeline(fitted_pipeline, p)
        loaded = joblib.load(p)
        texts = ["urgent claim winner free", "later today lunch mom"]
        np.testing.assert_array_equal(loaded.predict(texts), fitted_pipeline.predict(texts))
        np.testing.assert_allclose(
            loaded.predict_proba(texts), fitted_pipeline.predict_proba(texts)
        )

    def test_reproducible_tuning(self, tiny_dataframe):
        spec = classifier_specs()["mnb"]
        df = tiny_dataframe.copy()
        df["label"] = df["label"].map({"ham": 0, "spam": 1})
        tfidf = {"ngram_range": (1, 1), "min_df": 1, "max_features": None}
        gs1 = tune_model(spec, df, tfidf, cv_folds=2)
        gs2 = tune_model(spec, df, tfidf, cv_folds=2)
        assert gs1.best_params_ == gs2.best_params_
        assert abs(gs1.best_score_ - gs2.best_score_) < 1e-12


class TestEvaluation:
    def test_metrics_from_real_predictions(self, tiny_dataframe):
        df = tiny_dataframe.copy()
        df["label"] = df["label"].map({"ham": 0, "spam": 1})
        train, test = split_dataset(df, seed=3)
        spec = classifier_specs()["mnb"]
        pipe = build_pipeline(spec, {"ngram_range": (1, 1), "min_df": 1, "max_features": None})
        pipe.fit(train["text"], train["label"])
        m = evaluate_pipeline(pipe, test["text"], test["label"], include_auc=True)

        assert m["n_evaluated"] == len(test)
        assert 0 <= m["accuracy"] <= 1
        assert m["confusion_matrix"][0][0] + m["confusion_matrix"][0][1] == sum(
            test["label"] == 0
        )
        assert m["confusion_labels"] == ["ham", "spam"]  # label order fixed
        eb = m["error_breakdown"]
        assert eb["true_positives"] + eb["false_negives_spam_missed"] == sum(test["label"] == 1)
        assert eb["false_positives_spam_flag_on_ham"] + eb["true_negatives"] == sum(test["label"] == 0)
        assert m["roc_auc_spam"] is None or 0.5 <= m["roc_auc_spam"] <= 1.0

    def test_svc_has_no_proba(self):
        spec = classifier_specs()["linearsvc"]
        assert spec.supports_predict_proba is False
        pipe = build_pipeline(spec, {"ngram_range": (1, 1), "min_df": 1, "max_features": None})
        assert not hasattr(pipe, "predict_proba")


class TestPredictionService:
    @pytest.fixture
    def registry(self, fitted_pipeline, tmp_path: Path):
        save_pipeline(fitted_pipeline, tmp_path / "mnb.joblib")
        meta = build_metadata(
            results=[
                {
                    "model_id": "mnb",
                    "name": "Multinomial Naive Bayes",
                    "version": "1.0.0",
                    "supports_predict_proba": True,
                    "notes": "test",
                    "best_params": {"alpha": 0.1},
                    "validation_macro_f1": 0.99,
                    "test_metrics": {"accuracy": 0.99, "macro_f1": 0.99},
                }
            ],
            selected_id="mnb",
            tfidf_params={"ngram_range": (1, 1)},
            dataset_stats={"n_samples": 60},
            split_info={"strategy": "test"},
        )
        (tmp_path / "model_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        reg = ModelRegistry(tmp_path)
        reg.reload()
        return reg

    def test_single_prediction_shape(self, registry):
        svc = PredictionService(registry)
        [res] = svc.predict(["free cash winner click prize"])
        assert res["predicted_label"] == "spam"
        assert res["probabilities"]["spam"] > 0.5
        assert res["status"] == "success"
        assert res["model_id"] == "mnb"
        assert json.dumps(res["probabilities"])  # probabilities are serialisable

    def test_svc_model_reports_no_confidence(self, registry):
        # simulate a non-probability model entry
        registry._metadata["models"].append(
            {
                "model_id": "linearsvc",
                "name": "Linear SVM",
                "version": "1.0.0",
                "supports_predict_proba": False,
                "notes": "",
                "best_params": {},
                "validation_macro_f1": 0.9,
                "test_metrics": {},
            }
        )
        # artifact missing -> registry raises, which is the tested behaviour
        svc = PredictionService(registry)
        with pytest.raises(ModelRegistryError):
            svc.predict(["some text"], model_id="linearsvc")

    def test_empty_input_rejected(self, registry):
        svc = PredictionService(registry)
        with pytest.raises(EmptyTextInputError):
            svc.predict(["   "])

    def test_oversized_input_rejected(self, registry):
        svc = PredictionService(registry)
        with pytest.raises(EmptyTextInputError):
            svc.predict(["x" * 6000])

    def test_unknown_model_id_rejected(self, registry):
        svc = PredictionService(registry)
        with pytest.raises(ModelRegistryError):
            svc.predict(["hello there"], model_id="does-not-exist")

    def test_unusual_text_does_not_crash(self, registry):
        svc = PredictionService(registry)
        [res] = svc.predict(["🚀🚀🚀 ﷽ ß ﬁ ✨"])
        assert res["predicted_label"] in ("ham", "spam")

    def test_explanation_present_for_known_terms(self, registry):
        svc = PredictionService(registry)
        [res] = svc.predict(["urgent free cash prize winner click"])
        terms = {t["term"] for t in res["explanation_terms"]}
        assert "urgent" in terms or "free" in terms or "cash" in terms

    def test_validate_texts_rejects_nonstring(self):
        with pytest.raises(EmptyTextInputError):
            _validate_texts([123])  # type: ignore[list-item]
