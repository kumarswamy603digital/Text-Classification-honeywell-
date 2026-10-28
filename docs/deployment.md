# Deployment

Three ways to run Text Classification Studio, from quickest to most production-like.

---

## 1. Local development (two processes)

```powershell
# Backend
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m scripts.download_data
python -m scripts.explore_data
python -m scripts.train_models          # produces artifacts/
python -m uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev                             # http://localhost:5173
```

The frontend targets `http://localhost:8000` by default; override with
`VITE_API_BASE_URL` in `frontend/.env`. Backend CORS already allows
`http://localhost:5173` (see `cors_origins` in `.env.example`).

---

## 2. Docker Compose (full stack)

```bash
docker compose up --build
```

- **Frontend:** http://localhost:8080 — nginx serves the built SPA and proxies
  `/api/*` to the backend container (same-origin, so no CORS needed).
- **Backend:** http://localhost:8000/docs

On the first start the backend container checks for
`artifacts/model_metadata.json`; if absent it downloads the dataset and trains
all four models (a few minutes), then starts Uvicorn. Artifacts and data persist
in named volumes (`backend-artifacts`, `backend-data`) so restarts are instant.

### How the frontend reaches the API in Docker

The frontend image is built with `VITE_API_BASE_URL=""`, so the client uses the
relative base `/api/v1`. `frontend/nginx.conf` proxies `/api/` →
`http://backend:8000`. To point the SPA at an externally hosted API instead,
build with an explicit arg:

```bash
docker build --build-arg VITE_API_BASE_URL=https://api.example.com -t tcs-frontend ./frontend
```

### Configuration (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `CORS_ORIGINS` | `http://localhost:5173,...` | Allowed browser origins |
| `ARTIFACTS_DIR` | `artifacts` | Where trained pipelines/metadata live |
| `DATA_DIR` | `data` | Dataset location |
| `MAX_TEXT_LENGTH` | `5000` | Per-request text cap |
| `MAX_BATCH_SIZE` | `200` | Batch endpoint cap |
| `ENABLE_HISTORY` | `true` | Turn on SQLite prediction log |
| `HISTORY_DB_PATH` | `artifacts/history.db` | History database file |
| `HISTORY_STORE_TEXT` | `false` | Whether to persist raw message text |
| `RANDOM_SEED` | `42` | Reproducibility |

Copy `backend/.env.example` → `backend/.env` to override.

---

## 3. Manual/production notes

- **Serving:** run Uvicorn (or Uvicorn workers / Gunicorn with the Uvicorn worker
  class) behind a reverse proxy (nginx / a PaaS). The model registry loads once
  at startup in the app lifespan and is shared across requests.
- **Reproducibility:** artifacts are the product of `scripts/train_models.py`.
  Either bake them into the image or mount them via a volume/store. The API
  returns **503** until they exist — deploy a readiness probe against
  `/api/v1/ready`.
- **Retraining:** re-run `train_models.py` (or the container's first-run logic);
  call `registry.reload()` / restart the service to pick up new pipelines.
- **Stateless scaling:** predictions are stateless; the only stateful component is
  the optional SQLite history. For multi-replica deployments point
  `HISTORY_DB_PATH` at a shared DB or disable history.
- **Security:** error handlers deliberately strip stack traces and filesystem
  paths from responses. Set `CORS_ORIGINS` to your real front-end origin(s) in
  production. No secrets are committed; configure via environment only.

---

## Health & verification

```bash
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/ready
curl -X POST http://localhost:8000/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"text":"URGENT! You have won a FREE prize, call 09061234567"}'
```

A real response includes `predicted_label`, model provenance, and (for
probabilistic models) class probabilities.
