# Demo Script — Guided Walkthrough (~10 min)

A narrative for presenting the project. Start with both servers running
(backend on :8000, frontend on :5173) and models already trained.

> **Key message to land early:** every number and label in this UI is computed
> live by real fitted scikit-learn models over HTTP. Nothing is simulated.

---

## 0. Prove it's real (30s)

Terminal:
```bash
curl http://localhost:8000/api/v1/ready
```
→ `{"status":"ready","models_loaded":true,"selected_model_id":"linearsvc"}`

```bash
curl -X POST http://localhost:8000/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"text":"URGENT! You have won a FREE prize, call 09061234567 to claim"}'
```
→ real `spam` prediction with feature attributions.

---

## 1. Dashboard (`/`) — 1 min

- Task, dataset size and class balance (5,169 messages, ~6.9:1 ham:spam).
- **Selected model = Linear SVM**, chosen by validation macro-F1 (0.9701).
- Headline test metrics (accuracy 0.983, spam F1 0.929).
- Point out: these come straight from `/analytics/overview`, i.e. the artifacts
  written during training.

## 2. Classification Studio (`/studio`) — 3 min

1. Use an example chip ("likely spam") → **Classify message**.
2. Show the predicted label, class probabilities, and the **explanation panel**
   of top tokens (e.g. *free, win, claim, urgent*).
   - Stress: these are **feature attributions**, not causal proof.
3. Switch the model to **Linear SVM** and reclassify → note there is **no
   confidence bar**, replaced by an honest caveat that Linear SVM exposes no
   calibrated probabilities. *"We refuse to fabricate a fake confidence number."*
4. Switch to **Multinomial NB** → probabilities reappear. This contrast is the
   single best demonstration of the project's honesty principle.

## 3. Batch Classification (`/batch`) — 2 min

- Paste or upload a CSV. The parser auto-detects the text column.
- Run a batch (chunked to respect `max_batch_size`).
- Download results as CSV with label + confidence per row.
- Mention the dependency-free RFC-4180 parser (handles quotes, commas, newlines).

## 4. Model Comparison (`/compare`) — 2 min

- Side-by-side table across all 4 models: accuracy, spam precision/recall/F1,
  macro/weighted F1, ROC-AUC, and validation macro-F1.
- Confusion matrices + error breakdown (false positives vs missed spam).
- Explain the **selection protocol**: tuned via CV on train, selected by
  validation macro-F1, then evaluated **once** on a held-out test split —
  no test leakage.
- Note the differences are small and within noise given only 131 test spam
  examples → intellectual honesty about what the numbers do and don't show.

## 5. Dataset Explorer (`/dataset`) — 1 min

- Class counts, imbalance ratio, length distributions (spam ~2× longer than ham),
  top discriminative terms.
- Dataset provenance and limitations (dated 2007–09 corpus; demo of methodology,
  not a production filter).

## 6. About (`/about`) — 30s

- Architecture, tech stack, links to docs and the live Swagger at `/docs`.

---

## 7. Testing & reproducibility (closing) — 1 min

```bash
cd backend && python -m pytest        # 55 tests pass
cd frontend && npm run test           # 19 tests pass
```

- Emphasise: unit tests for the ML helpers, integration tests that train tiny
  models and drive the whole API, frontend tests that mock the client and assert
  real request/response wiring.
- One command (`docker compose up --build`) reproduces the entire system,
  including dataset download and training, in containers.

---

## Anticipated questions

- **"Why no deep learning / transformers?"** The brief is traditional ML; TF-IDF
  + linear models are strong, fast, interpretable, and reproducible on CPU.
- **"Which metric should I trust?"** Not accuracy (87% is free by predicting ham).
  Spam precision/recall/F1 and macro-F1, on a held-out split.
- **"Can I trust a single prediction?"** It's a heuristic; low-confidence cases
  are flagged for review. Explanations are attributions, not reasons.
- **"How do you avoid train/serve skew?"** The full preprocessing+vectorizer+model
  is one serialised pipeline artifact.
