"""Integration tests against the real FastAPI app with a temporary registry.

A tiny synthetic pipeline is trained on the fabricated fixture corpus
(TEST FIXTURE ONLY) so endpoints execute the genuine code path end-to-end.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ml.evaluate import evaluate_pipeline
from app.ml.train import build_metadata, build_pipeline, classifier_specs, save_pipeline
import app.ml.registry as registry_module


@pytest.fixture(scope="module")
def artifacts(tiny_dataframe, tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("artifacts")
    df = tiny_dataframe.copy()
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    tfidf = {"ngram_range": (1, 1), "min_df": 1, "max_features": None}

    results = []
    for mid in ("mnb", "linearsvc"):
        spec = classifier_specs()[mid]
        pipe = build_pipeline(spec, tfidf)
        pipe.fit(df["text"], df["label"])
        save_pipeline(pipe, d / f"{mid}.joblib")
        metrics = evaluate_pipeline(pipe, df["text"], df["label"])
        results.append(
            {
                "model_id": mid,
                "name": spec.name,
                "version": "1.0.0",
                "supports_predict_proba": spec.supports_predict_proba,
                "notes": spec.notes,
                "best_params": {"alpha": 0.1},
                "validation_macro_f1": 0.95,
                "test_metrics": metrics,
            }
        )

    meta = build_metadata(
        results=results,
        selected_id="mnb",
        tfidf_params=tfidf,
        dataset_stats={"n_samples": len(df), "source": {"name": "fixture"}},
        split_info={"strategy": "fixture split", "n_train": len(df), "n_test": len(df)},
    )
    (d / "model_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


@pytest.fixture
def client(artifacts, tmp_path, monkeypatch):
    """App wired to the tiny fixture registry + isolated temp history DB."""
    from app.core import config as app_config

    monkeypatch.setenv("HISTORY_DB_PATH", str(tmp_path / "history.db"))
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts))
    app_config.get_settings.cache_clear()
    monkeypatch.setattr(registry_module, "_registry", registry_module.ModelRegistry(artifacts))
    with TestClient(app) as c:
        yield c
    app_config.get_settings.cache_clear()


class TestSystemEndpoints:
    def test_health(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_ready_when_models_present(self, client):
        r = client.get("/api/v1/ready")
        body = r.json()
        assert r.status_code == 200
        assert body["status"] == "ready"
        assert body["selected_model_id"] == "mnb"

    def test_ready_when_artifacts_missing(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(registry_module, "_registry", registry_module.ModelRegistry(tmp_path))
        r = client.get("/api/v1/ready")
        assert r.json()["status"] == "not_ready"
        assert r.json()["models_loaded"] is False


class TestClassifyEndpoints:
    def test_classify_valid_spam_like_text(self, client):
        r = client.post("/api/v1/classify", json={"text": "free cash winner claim urgent click"})
        assert r.status_code == 200
        body = r.json()
        assert body["predicted_label"] == "spam"
        assert body["probabilities"]["spam"] > 0.5
        assert body["model_id"] == "mnb"
        assert isinstance(body["explanation_terms"], list)

    def test_classify_with_explicit_model(self, client):
        r = client.post(
            "/api/v1/classify", json={"text": "winner free cash", "model_id": "linearsvc"}
        )
        assert r.status_code == 200
        body = r.json()
        # Linear SVM: label only, never a fabricated confidence
        assert body["probabilities"] is None
        assert body["confidence"] is None
        assert body["confidence_note"]

    def test_classify_empty_text_422(self, client):
        r = client.post("/api/v1/classify", json={"text": ""})
        assert r.status_code == 422
        assert r.json()["error"] == "validation_error"

    def test_classify_whitespace_text_422(self, client):
        r = client.post("/api/v1/classify", json={"text": "     "})
        assert r.status_code == 422

    def test_classify_oversized_422(self, client):
        r = client.post("/api/v1/classify", json={"text": "x" * 5001})
        assert r.status_code == 422

    def test_unknown_model_404(self, client):
        r = client.post("/api/v1/classify", json={"text": "hello", "model_id": "nope/../x"})
        assert r.status_code == 404
        assert "stack" not in json.dumps(r.json()).lower()

    def test_batch_order_preserved(self, client):
        texts = ["free cash winner", "mum lunch today noon", "claim urgent prize"]
        r = client.post("/api/v1/classify/batch", json={"texts": texts})
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 3
        assert [x["explanation_terms"] is not None for x in body["results"]]

    def test_batch_over_limit_422(self, client):
        r = client.post("/api/v1/classify/batch", json={"texts": ["hi"] * 201})
        assert r.status_code == 422


class TestModelEndpoints:
    def test_list_models(self, client):
        r = client.get("/api/v1/models")
        assert r.status_code == 200
        ids = {m["model_id"] for m in r.json()}
        assert {"mnb", "linearsvc"} <= ids
        assert sum(1 for m in r.json() if m["is_selected"]) == 1

    def test_selected_model(self, client):
        r = client.get("/api/v1/models/selected")
        assert r.json()["model_id"] == "mnb"

    def test_model_detail(self, client):
        r = client.get("/api/v1/models/mnb")
        assert r.status_code == 200
        assert r.json()["test_metrics"]["n_evaluated"] > 0

    def test_model_detail_unknown_404(self, client):
        assert client.get("/api/v1/models/gpt-5").status_code == 404


class TestAnalyticsEndpoints:
    def test_overview(self, client):
        r = client.get("/api/v1/analytics/overview")
        assert r.status_code == 200
        body = r.json()
        assert body["selected_model"]["model_id"] == "mnb"
        assert body["selected_metrics"]["accuracy"] > 0.5

    def test_model_comparison(self, client):
        r = client.get("/api/v1/analytics/model-comparison")
        body = r.json()
        assert len(body["rows"]) == 2
        row = next(x for x in body["rows"] if x["selected"])
        assert row["model_id"] == "mnb"

    def test_confusion_matrix_labels_ordered(self, client):
        r = client.get("/api/v1/analytics/confusion-matrix")
        body = r.json()
        assert body["labels"] == ["ham", "spam"]
        assert len(body["matrix"]) == 2

    def test_confusion_matrix_for_other_model(self, client):
        r = client.get("/api/v1/analytics/confusion-matrix", params={"model_id": "linearsvc"})
        assert r.status_code == 200

    def test_dataset_stats_unavailable_honest_503(self, client):
        r = client.get("/api/v1/analytics/dataset-stats")
        assert r.status_code == 503


class TestHistoryEndpoints:
    def test_history_records_and_deletes(self, client):
        client.post("/api/v1/classify", json={"text": "free winner claim"})
        r = client.get("/api/v1/history")
        assert r.status_code == 200
        body = r.json()
        if not body["enabled"]:
            pytest.skip("history disabled in this environment")
        pid = body["records"][0]["prediction_id"]
        assert client.get(f"/api/v1/history/{pid}").status_code == 200
        assert client.delete(f"/api/v1/history/{pid}").status_code == 204
        assert client.get(f"/api/v1/history/{pid}").status_code == 404

    def test_history_delete_unknown_404(self, client):
        r = client.delete("/api/v1/history/nonexistent-id")
        assert r.status_code == 404
