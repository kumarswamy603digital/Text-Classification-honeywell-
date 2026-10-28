# Architecture

## Overview

Text Classification Studio is a two-tier application:

```
┌────────────────────┐        HTTP/JSON        ┌─────────────────────────┐
│  React SPA         │  ───────────────────▶   │  FastAPI backend        │
│  (Vite + TS)       │  ◀───────────────────   │  /api/v1/*              │
│                    │       typed client      │                         │
│  Dashboard         │                          │  ModelRegistry          │
│  Studio            │                          │   └─ Pipeline(s).joblib │
│  Batch             │                          │  PredictionService      │
│  Comparison        │                          │  AnalyticsService       │
│  Dataset Explorer  │                          │  HistoryService (SQLite)│
│  About             │                          │  Scripts (train/eval)   │
└────────────────────┘                          └─────────────────────────┘
```

The frontend never computes a label itself. Every prediction, metric and chart
is fetched from the backend, which serves **actual fitted scikit-learn
pipelines**.

## Backend layers

| Layer | Location | Responsibility |
|---|---|---|
| Config | `app/core/config.py` | Env-driven settings (`Settings`, cached via `lru_cache`) |
| ML primitives | `app/ml/` | dataset load/validate, preprocessing, TF-IDF, training, evaluation, prediction, registry |
| Schemas | `app/schemas/api_schemas.py` | Pydantic request/response contracts (single source of truth for the API shape) |
| Services | `app/services/` | analytics aggregation, optional SQLite history |
| API | `app/api/routes.py` | Thin HTTP layer, dependency injection |
| App | `app/main.py` | Lifespan (load registry), CORS, centralised exception handlers |

### The pipeline (anti-drift design)

Each model is a single scikit-learn `Pipeline`:

```
TextPreprocessor  →  TfidfVectorizer  →  Classifier
 (custom transformer)   (word 1–2 grams)   (NB / LR / LinearSVC / CNB)
```

The **whole pipeline is serialised to one `.joblib` file**. At inference the same
object is loaded, so preprocessing/feature-extraction can never diverge between
training and serving — the classic source of silent model/production skew.

### Model registry

`app/ml/registry.py` loads `artifacts/model_metadata.json` at startup and
lazily, thread-safely materialises each pipeline. It exposes:

- `is_ready()` — are artifacts present and loadable?
- `get_model_info(model_id)` — validated metadata for a model
- `get_pipeline(model_id)` — the fitted pipeline
- `reload()` — refresh after retraining

If artifacts are missing, the registry raises `ModelRegistryError`, which the API
maps to **HTTP 503** with a clean message (no stack trace or filesystem paths).

### Training & selection protocol

`scripts/train_models.py`:

1. Download/validate the dataset; remove exact duplicates.
2. Stratified 80/20 split (seed 42). The **test split is set aside and not used
   until final evaluation**.
3. Tune the TF-IDF representation on the training data (CV).
4. For each of the 4 classifiers, `GridSearchCV` (3-fold stratified) over a small
   hyper-parameter grid, scored by **macro-F1**.
5. Select the model with the best **validation macro-F1**.
6. Train the winner's full pipeline on the train split, then evaluate **once** on
   the held-out test split.
7. Persist pipelines, `model_metadata.json`, `dataset_stats.json`, and
   `model_metrics.csv`.

### Prediction & explanations

`app/ml/predict.py` returns the discrete label, class probabilities (only when
the estimator supports `predict_proba`), and top feature attributions derived
from:

- linear models → `coef_` rows of the classifier step;
- Naive Bayes variants → ratio of `feature_log_prob_` across classes.

**Honesty rule:** Linear SVM has no calibrated probabilities. The API returns
`probabilities = null` and `confidence = null` with an explanatory note rather
than squashing a raw decision margin into a pseudo-probability.

## Frontend layers

| Concern | Location | Notes |
|---|---|---|
| API contract | `src/types/api.ts` | Mirrors backend Pydantic schemas |
| Client | `src/api/client.ts` | Single typed `fetch` wrapper, `ApiError`, base URL from env |
| Data hook | `src/hooks/useApi.ts` | Minimal loading/error/retry state — no external store needed |
| UI primitives | `src/components/ui/` | Accessible Card/Button/Badge + a keyboard-navigable Select built on `@floating-ui/react` |
| Feature components | `src/components/analytics/`, `src/components/classification/` | Recharts visualisations and prediction widgets |
| Pages | `src/pages/` | 6 routes wired through React Router |

State is intentionally lightweight: pages call `api.*` through `useApi`, so the
UI always reflects live backend responses and shows explicit loading / error /
not-ready states.

## Error handling & graceful degradation

- **Backend:** centralised handlers convert `RequestValidationError` → 422,
  `ModelRegistryError` → 503, unexpected errors → 500 with a generic message.
  Internal paths and stack traces are never leaked to clients.
- **Frontend:** every page handles `loading`, `error` (including network-unreachable
  → "Is the API running on port 8000?") and empty/not-ready states, so the app is
  informative before models are trained or while the backend is down.

## Testing strategy

- **Backend:** pure-unit tests for preprocessing/dataset/feature/train helpers, plus
  integration tests that train tiny models in a temp artifacts dir and drive the
  full API through FastAPI's `TestClient` (health/ready, classify valid/empty/
  oversize/unknown-model, batch, analytics, history).
- **Frontend:** Vitest + React Testing Library — CSV parser/format utilities and a
  Studio page test with the API client mocked, asserting real request/response
  wiring (selected-model default, classify call, non-probability caveat, error
  surfacing, disabled empty submit).
