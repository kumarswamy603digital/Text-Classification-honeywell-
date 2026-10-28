# API Reference

Base path: **`/api/v1`** · Interactive docs: **`/docs`** (Swagger UI) and **`/redoc`**.
All request/response bodies are JSON. Validation errors return **422**; a missing
model registry returns **503**; unexpected errors return **500** with a generic
message (no stack traces or filesystem paths are leaked).

Common limits (configurable via env): `max_text_length = 5000`,
`max_batch_size = 200`.

---

## Liveness & readiness

### `GET /health`
```json
{ "status": "ok", "app": "Text Classification Studio API", "version": "1.0.0" }
```

### `GET /ready`
Are models loaded and can predictions be served?
```json
{ "status": "ready", "models_loaded": true, "selected_model_id": "linearsvc", "detail": null }
```
When artifacts are absent: `status: "not_ready"` with an explanatory `detail`.

---

## Classification

### `POST /classify`
Classify a single message with a real fitted pipeline.

Request:
```json
{ "text": "URGENT! You have won a FREE prize, call 09061234567", "model_id": "mnb" }
```
- `text` — required, 1–5000 chars (empty → 422; too long → 422).
- `model_id` — optional; defaults to the **selected** model. Unknown id → 404.

Response (`ClassifyResponse`):
```json
{
  "prediction_id": "3644b1c3-...",
  "model_id": "linearsvc",
  "model_name": "Linear SVM",
  "model_version": "1.0.0",
  "predicted_label": "spam",
  "label_display": "Spam",
  "probabilities": null,
  "confidence": null,
  "confidence_type": null,
  "confidence_note": "Linear SVM does not expose calibrated class probabilities; no confidence figure is reported rather than a converted score.",
  "explanation_terms": [ { "term": "call", "weight": 1.08 }, { "term": "free", "weight": 1.06 } ],
  "latency_ms": 1.57,
  "status": "success",
  "timestamp": "2026-10-27T17:09:22.959536+00:00"
}
```

Honesty notes:
- **Probabilistic models** (`mnb`, `logreg`, `cnb`) return `probabilities` as
  `{ "ham": p, "spam": p }` and `confidence` = probability of the predicted class.
- **Linear SVM** returns `probabilities = null` and `confidence = null` with an
  explanatory `confidence_note` — no pseudo-probability is fabricated.
- `explanation_terms` are **feature attributions**, not causal evidence.

### `POST /classify/batch`
Request:
```json
{ "texts": ["...", "..."], "model_id": null }
```
Response:
```json
{ "results": [ /* ClassifyResponse[] */ ], "count": 2, "model_id": "linearsvc" }
```
Exceeding `max_batch_size` → 413/422.

---

## Models

### `GET /models` → `ModelSummary[]`
```json
[ { "model_id": "mnb", "name": "Multinomial Naive Bayes", "version": "1.0.0",
    "supports_predict_proba": true, "notes": "...", "is_selected": false } ]
```

### `GET /models/selected` → `ModelSummary`
The model the API uses by default.

### `GET /models/{model_id}` → `ModelDetail`
Adds `best_params`, `validation_macro_f1`, and full `test_metrics`
(accuracy, per-class precision/recall/F1, macro/weighted F1, ROC-AUC/PR-AUC when
available, `confusion_matrix`, `error_breakdown`).

---

## Analytics

| Endpoint | Returns |
|---|---|
| `GET /analytics/overview` | Dashboard aggregate: task, dataset stats, selected model + metrics, `n_models_trained`, `history_enabled`, `trained_at`, `selection_criterion` |
| `GET /analytics/model-comparison` | `{ protocol, rows[] }` — one metric row per model plus the selection protocol description |
| `GET /analytics/confusion-matrix?model_id=` | `{ model_id, labels, matrix, n_evaluated, positive_class, error_breakdown }` |
| `GET /analytics/dataset-stats` | Full dataset statistics (counts, balance, imbalance ratio, text-length distribution, top terms, source) |

`/analytics/*` and prediction endpoints return **503** until models are trained.

---

## History (optional, SQLite)

Enabled when `enable_history = true`. Records only metadata by default
(`history_store_text = false` stores no message text).

| Endpoint | Purpose |
|---|---|
| `GET /history?limit=20` | `{ records[], total, enabled }` recent predictions |
| `GET /history/{prediction_id}` | Single record |
| `DELETE /history/{prediction_id}` | Remove a record (204) |

---

## Error model

| Status | Meaning |
|---|---|
| 404 | Unknown `model_id` or history id |
| 413 / 422 | Batch too large / validation error (empty, oversized text, bad body) |
| 503 | Model registry not ready (artifacts missing) — retry after training |
| 500 | Unexpected server error (generic message only) |

FastAPI validation errors return an array in `detail`; the frontend client
flattens `detail[].msg` into a readable string.
