# Model Report — SMS Spam Detection

> All figures below are **measured**, produced by `scripts/train_models.py`, and
> stored in `backend/artifacts/model_metadata.json` / `model_metrics.csv`. They
> are read live by the API and frontend — never hardcoded. Positive class = **spam**.

## Protocol

1. Load + validate the UCI SMS Spam Collection; remove 403 exact duplicates →
   **5,169** rows (4,516 ham / 653 spam).
2. Stratified **80/20** train/test split (seed 42): 4,135 train / 1,034 test.
3. Tune the TF-IDF representation on the training data via cross-validation.
4. For each candidate classifier, `GridSearchCV` (3-fold stratified) scored by
   **macro-F1**.
5. Select the model with the best **validation macro-F1**.
6. Refit the winner on train and evaluate **once** on the held-out test split.

### Chosen TF-IDF configuration

| Parameter | Value |
|---|---|
| `ngram_range` | (1, 2) — word unigrams + bigrams |
| `min_df` | 2 |
| `max_df` | 0.95 |
| `sublinear_tf` | True |
| `max_features` | 50,000 |
| char n-grams | disabled |

## Models compared

| model_id | Estimator | predict_proba? | Tuned param (best) | Val macro-F1 |
|---|---|---|---|---|
| `mnb` | Multinomial Naive Bayes | ✅ | `alpha = 0.05` | 0.9647 |
| `logreg` | Logistic Regression | ✅ | `C = 10.0` | 0.9627 |
| **`linearsvc`** | **Linear SVM** ⭐ selected | ❌ | `C = 4.0` | **0.9701** |
| `cnb` | Complement Naive Bayes | ✅ | `alpha = 1.0` | 0.9547 |

⭐ **Selected model: Linear SVM** — highest validation macro-F1 (0.9701).

## Held-out test results (n = 1,034, evaluated once)

| Model | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) | Macro-F1 | Weighted-F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| Multinomial NB | 0.9836 | 0.9672 | 0.9008 | 0.9328 | 0.9617 | 0.9833 | 0.9909 | 0.9731 |
| Logistic Regression | 0.9807 | 0.9744 | 0.8702 | 0.9194 | 0.9542 | 0.9802 | 0.9929 | 0.9749 |
| **Linear SVM (selected)** | 0.9826 | 0.9593 | 0.9008 | 0.9291 | 0.9596 | 0.9824 | n/a¹ | n/a |
| Complement NB | 0.9826 | 0.9669 | 0.8931 | 0.9286 | 0.9593 | 0.9823 | 0.9832 | 0.9615 |

¹ Linear SVM does not expose calibrated probabilities, so ranking metrics that
need a score/probability (ROC-AUC/PR-AUC) are intentionally **not reported**
rather than computed from an uncalibrated decision margin and presented as if
they were probabilities.

## Confusion matrices (test split)

Rows = actual, columns = predicted, order `[ham, spam]`.

| Model | TN (ham→ham) | FP (ham→spam) | FN (spam→ham) | TP (spam→spam) |
|---|---|---|---|---|
| Multinomial NB | 899 | 4 | 13 | 118 |
| Logistic Regression | 900 | 3 | 17 | 114 |
| Linear SVM (selected) | 898 | 5 | 13 | 118 |
| Complement NB | 899 | 4 | 14 | 117 |

Test split class support: 903 ham, 131 spam.

## Interpretation

- All four models are strong and **statistically close** on this corpus; the
  spam-side differences (recall 0.87–0.90) sit on only 131 positive examples, so
  sub-1% gaps are within noise and should not be over-read.
- **Logistic Regression** is the most conservative about false alarms (only 3
  ham flagged as spam) but misses the most spam (recall 0.870).
- **Multinomial NB** and the **selected Linear SVM** give the best spam recall
  (0.901) at the cost of a couple more false positives.
- Selection by **macro-F1** balances both classes rather than chasing the ~98%
  accuracy that the 87%-ham majority alone would deliver.
- **Operating point matters:** the report includes a balanced threshold per
  probabilistic model (e.g. NB/LR threshold ≈ 0.27–0.29 equalises precision and
  recall near 0.92–0.94). Deployments that prize low false-positive rates should
  raise the threshold; spam-blocking that prizes catch-rate should lower it.

## Explanation caveats

The UI shows top per-token weights ("feature attributions") to explain which
words pushed a message toward spam or ham. These are **attributional cues from a
linear model, not causal proof** — high-weight tokens like *free / win / claim /
prize / urgent* and digit strings reflect correlation in the training corpus.
Predictions are probabilistic/heuristic outputs and should not be treated as a
guarantee of a message's true intent.

## Reproducing

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m scripts.train_models
python -m scripts.evaluate_models   # re-prints the metrics table from artifacts
```
