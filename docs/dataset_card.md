# Dataset Card — SMS Spam Collection

## Intended use

Binary text classification: flag an SMS message as **spam** (unsolicited
commercial/promotional text) or **ham** (legitimate). Used here as the default
task to exercise the full ML lifecycle with real, reproducible results.

## Provenance

- **Source:** [UCI SMS Spam Collection, dataset 228](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Original citation:** A. Almeida, J. Gómez Hidalgo, A. Yamakura, *Contributions to the Study of SMS Spam Filtering*, 2011.
- **License / terms:** Provided for general and educational use under UCI's terms of access.
- **Acquisition:** `backend/scripts/download_data.py` downloads the official zip
  with Python's standard library, extracts `SMSSpamCollection`, and writes
  `backend/data/raw/`. If the download fails it raises — the pipeline **never
  fabricates substitute data**.

## Labels & mapping

| Raw label | Mapped id | Meaning |
|---|---|---|
| `ham` | 0 | Legitimate message |
| `spam` | 1 | Unsolicited / promotional message (**positive class**) |

`validate_dataset` rejects any unexpected label value and drops rows whose label
cannot be parsed (logged), then casts labels to `int64`.

## Composition (after cleaning)

| Metric | Value |
|---|---|
| Working samples | **5,169** |
| Ham | 4,516 (87.4%) |
| Spam | 653 (12.6%) |
| Imbalance ratio | ~6.9 : 1 (ham : spam) |
| Exact duplicates removed | 403 |

> The public file is commonly quoted as 5,574 messages. This project removes exact
> duplicate (label, text) pairs before modelling, yielding 5,169 rows.

## Text length (a real signal in this dataset)

| | min | median | mean | max |
|---|---|---|---|---|
| Ham (chars) | 2 | 53 | 70.9 | 910 |
| Spam (chars) | 13 | 148 | 137.7 | 223 |

Spam messages are markedly longer (mean ~138 vs ~71 chars) — spam SMS cluster
near the operator length limits, whereas ham includes very short replies. This
length gap, plus lexical cues (e.g. *free, win, claim, prize, urgent, call* +
phone numbers), is what the TF-IDF models learn.

## Split (reproducibility)

- **Strategy:** stratified 80/20 train/test, `random_state = 42`
- **Train:** 4,135 rows (before CV)
- **Test:** 1,034 rows — held out and evaluated **once**, after model selection
- Hyper-parameter tuning uses 3-fold **stratified CV on the train split only**.

## Preprocessing

`app/ml/preprocessing.py` (a custom pipeline transformer) applies light,
language-agnostic normalisation (whitespace/URL/email/number handling and case
folding) before TF-IDF. It is stored inside the saved pipeline, so serving reuses
exactly the same transformation.

## Known limitations & caveats

- **Class imbalance (~6.9:1):** raw accuracy is optimistic about spam detection;
  the project therefore reports precision/recall/F1 for the **spam** class and
  macro-F1, and selects models by validation **macro-F1**, not accuracy.
- **Small spam subset (653):** spam-side metrics carry noticeable variance; treat
  differences of <1% between models as not meaningful.
- **Dated corpus:** collected circa 2007–09; modern spam/phrasing may differ, so
  this is a demonstration of methodology, not a production spam filter.
- **No PII handling required** beyond the standard dataset; messages are
  short SMS texts.

## Regenerating

```powershell
cd backend
python -m scripts.download_data     # fetch + validate
python -m scripts.explore_data      # EDA summary
python -m scripts.train_models      # produces artifacts/dataset_stats.json
```

Statistics shown in the UI come from `backend/artifacts/dataset_stats.json`,
written during training — never hardcoded.
