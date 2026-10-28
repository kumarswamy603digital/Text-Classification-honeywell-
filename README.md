# Text Classification Studio

A complete, **runnable** end-to-end machine-learning system that classifies SMS
messages as **spam** or **ham** (legitimate). Every prediction you see in the UI
is produced by a **real fitted scikit-learn pipeline** served by a FastAPI
backend — nothing is mocked, simulated, or hardcoded.

Built to demonstrate the full ML lifecycle for an AI/ML internship:

> data acquisition → EDA → preprocessing → feature extraction (TF-IDF) →
> training multiple models → hyper-parameter tuning → model selection →
> honest held-out evaluation → serving via API → interactive React frontend →
> automated tests → containerised deployment.

---

## At a glance (real, measured results)

- **Task:** SMS spam detection (binary: `ham` / `spam`, positive class = `spam`)
- **Dataset:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Working set after cleaning:** 5,169 messages (4,516 ham / 653 spam, ~6.9:1 imbalance) after removing 403 exact duplicates
- **Split:** stratified 80/20 (train 4,135 / test 1,034), seed = 42
- **Selected model:** Linear SVM (chosen by **validation macro-F1 = 0.9701**)

Held-out **test** metrics for all four models (evaluated once, after selection):

| Model | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) | Macro-F1 |
|---|---|---|---|---|---|
| Multinomial NB | 0.9836 | 0.9672 | 0.9008 | 0.9328 | 0.9617 |
| Logistic Regression | 0.9807 | 0.9744 | 0.8702 | 0.9194 | 0.9542 |
| **Linear SVM (selected)** | **0.9826** | 0.9593 | 0.9008 | 0.9291 | 0.9596 |
| Complement NB | 0.9826 | 0.9669 | 0.8931 | 0.9286 | 0.9593 |

These numbers are regenerated every time you run training; they are read from
`backend/artifacts/model_metadata.json` at runtime and are **never hardcoded** in
the UI. See [`docs/model_report.md`](docs/model_report.md) for the full report.

---

## Tech stack

**Backend:** Python · FastAPI · Pydantic v2 · pandas · NumPy · SciPy · scikit-learn · joblib · Uvicorn

**Frontend:** React 19 · Vite · TypeScript · Tailwind CSS v4 · React Router v7 · Recharts · @floating-ui/react

**Testing:** pytest + FastAPI TestClient (backend) · Vitest + React Testing Library (frontend)

**Core classification uses only traditional ML** — TF-IDF features plus
Multinomial NB, Logistic Regression, Linear SVM, and Complement NB. No
transformers, neural networks, or LLMs.

---

## Project layout

```
text classification/
├── backend/
│   ├── app/
│   │   ├── api/routes.py          # FastAPI endpoints
│   │   ├── core/config.py         # Env-driven settings
│   │   ├── ml/                    # datasets, preprocessing, features,
│   │   │                          # train, evaluate, predict, registry
│   │   ├── schemas/api_schemas.py # Pydantic request/response models
│   │   ├── services/              # analytics + SQLite history
│   │   └── main.py                # App factory, lifespan, CORS, error handlers
│   ├── scripts/                   # download_data, explore_data, train_models, evaluate_models
│   ├── tests/                     # pytest unit + integration suites
│   ├── artifacts/                 # trained pipelines + metrics (generated)
│   ├── data/                      # raw + processed dataset (generated)
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── api/client.ts          # Typed API client
│   │   ├── components/            # ui, analytics, classification, layout
│   │   ├── pages/                 # Dashboard, Studio, Batch, Comparison, DatasetExplorer, About
│   │   ├── hooks/, types/, utils/
│   │   └── App.tsx, main.tsx
│   ├── tests/                     # Vitest suites
│   ├── package.json
│   └── Dockerfile
├── docs/                          # architecture, dataset card, model report, API ref, deployment, demo
├── notebooks/                     # exploratory data analysis notebook
├── docker-compose.yml
└── README.md
```

---

## Quick start (local)

### 1. Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt

# Download the dataset and run EDA
python -m scripts.download_data
python -m scripts.explore_data

# Train all 4 models, tune, select, evaluate, and save artifacts
python -m scripts.train_models

# Start the API (http://127.0.0.1:8000, docs at /docs)
python -m uvicorn app.main:app --reload --port 8000
```

### 2. Frontend

```powershell
cd frontend
npm install
npm run dev                         # http://localhost:5173
```

Open **http://localhost:5173** — the dashboard, studio, batch, comparison and
dataset pages all read live data from the backend.

> The frontend talks to `http://localhost:8000` by default. Override with
> `VITE_API_BASE_URL` (see `frontend/.env.example`).

---

## Running the tests

```powershell
# Backend (55 tests)
cd backend
.\.venv\Scripts\python.exe -m pytest

# Frontend (19 tests)
cd frontend
npm run test
```

---

## Running with Docker

```bash
docker compose up --build
```

- Frontend: http://localhost:5173 (Vite dev) or via the nginx image on http://localhost:8080
- Backend:  http://localhost:8000/docs

See [`docs/deployment.md`](docs/deployment.md).

---

## API surface

All endpoints are under `/api/v1`. Interactive docs at `/docs` (Swagger) and `/redoc`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/ready` | Are models loaded? |
| POST | `/classify` | Classify a single message |
| POST | `/classify/batch` | Classify many messages |
| GET | `/models` | List trained models |
| GET | `/models/selected` | The selected (default) model |
| GET | `/models/{model_id}` | Model detail + test metrics |
| GET | `/analytics/overview` | Dashboard summary |
| GET | `/analytics/model-comparison` | All-model comparison |
| GET | `/analytics/confusion-matrix` | Confusion matrix (per model) |
| GET | `/analytics/dataset-stats` | Dataset statistics |
| GET/DELETE | `/history`, `/history/{id}` | Optional SQLite prediction log |

Full request/response schemas in [`docs/api_reference.md`](docs/api_reference.md).

---

## Design principles

- **Prevent train/inference drift:** each model is a single
  `Pipeline(preprocessor → TfidfVectorizer → classifier)` saved as one `.joblib`
  artifact, so preprocessing can never diverge between training and serving.
- **Honest evaluation:** tuning/selection use validation macro-F1; the test split
  is touched **once**, after selection. Metrics are read from artifacts, never
  hardcoded.
- **Honest confidence:** Linear SVM has no calibrated probabilities, so the API
  deliberately reports **no confidence** for it rather than converting a raw
  decision score into a fake probability.
- **Explanations, not causation:** per-token weights are shown as *feature
  attributions* (which tokens pushed the decision), clearly labelled as such.
- **Graceful degradation:** if artifacts are missing the API returns 503 and the
  UI shows a clear "train the models first" state instead of crashing.

---

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — system design and data flow
- [`docs/dataset_card.md`](docs/dataset_card.md) — dataset provenance, composition, caveats
- [`docs/model_report.md`](docs/model_report.md) — training protocol and full results
- [`docs/api_reference.md`](docs/api_reference.md) — endpoint contract
- [`docs/deployment.md`](docs/deployment.md) — Docker and production notes
- [`docs/demo_script.md`](docs/demo_script.md) — guided walkthrough

## License / data terms

The UCI SMS Spam Collection is provided for general and educational use under
UCI's terms of access. This project is an educational internship artefact.
