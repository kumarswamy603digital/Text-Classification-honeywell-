"""Dataset acquisition, validation and EDA statistics.

Default task: SMS spam detection using the UCI SMS Spam Collection
(https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

The dataset is loaded from a local file when present; otherwise the
acquisition script (scripts/download_data.py) fetches the official zip.
We NEVER fabricate training data if the download fails.
"""
from __future__ import annotations

import io
import json
import logging
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
INTERNAL_FILE = "SMSSpamCollection"

# Canonical label mapping for the SMS spam task
LABEL_MAP = {"ham": 0, "spam": 1}
CLASS_NAMES = ["ham", "spam"]  # index == label value


class DatasetError(RuntimeError):
    """Raised when the dataset is missing, unreadable or fails validation."""


def download_raw(data_dir: Path, force: bool = False) -> Path:
    """Download the UCI SMS Spam Collection zip and extract the raw TSV.

    Returns the path to the extracted `SMSSpamCollection` file.
    Raises DatasetError on any network / integrity failure.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / INTERNAL_FILE
    if target.exists() and not force:
        logger.info("Raw dataset already present at %s", target)
        return target

    logger.info("Downloading dataset from %s ...", UCI_ZIP_URL)
    try:
        with urllib.request.urlopen(UCI_ZIP_URL, timeout=60) as resp:
            blob = resp.read()
    except Exception as exc:  # noqa: BLE001
        raise DatasetError(
            f"Failed to download the dataset: {exc}. "
            "Place the 'SMSSpamCollection' file manually into the data directory."
        ) from exc

    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            if INTERNAL_FILE not in zf.namelist():
                raise DatasetError(
                    f"'{INTERNAL_FILE}' not found inside the downloaded archive; "
                    f"archive contained: {zf.namelist()}"
                )
            target.write_bytes(zf.read(INTERNAL_FILE))
    except zipfile.BadZipFile as exc:
        raise DatasetError("Downloaded archive is not a valid zip file") from exc

    logger.info("Raw dataset written to %s (%d bytes)", target, target.stat().st_size)
    return target


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load and validate the labelled dataset into columns [label, text].

    Accepts the whitespace/tab-separated UCI raw file or a labelled CSV with
    'label'/'text' (or 'v1'/'v2') columns.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise DatasetError(
            f"Dataset file not found at {data_path}. "
            "Run `python scripts/download_data.py` first."
        )

    try:
        sample = data_path.read_text(encoding="utf-8", errors="replace")[:4096]
    except OSError as exc:
        raise DatasetError(f"Dataset file is not readable: {exc}") from exc

    first_line = sample.splitlines()[0] if sample.splitlines() else ""
    try:
        if "," in first_line and not first_line.startswith(("ham", "spam")):
            # labelled CSV variant
            df = _read_csv_flexible(data_path)
        elif "\t" in sample:
            df = pd.read_csv(data_path, sep="\t", header=None, names=["label", "text"])
        else:
            # UCI raw file is whitespace separated: label, then free text
            df = pd.read_csv(
                data_path, sep=r"\s+", header=None, names=["label", "text"], engine="python"
            )
    except DatasetError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DatasetError(f"Failed to parse dataset: {exc}") from exc

    return validate_dataset(df)


def _read_csv_flexible(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    cols = {c.lower().strip(): c for c in df.columns}
    label_col = next((cols[k] for k in ("label", "v1", "target", "class") if k in cols), None)
    text_col = next((cols[k] for k in ("text", "message", "v2", "body") if k in cols), None)
    if label_col is None or text_col is None:
        raise DatasetError(
            f"CSV must contain a label column and a text column; found {list(df.columns)}"
        )
    return df[[label_col, text_col]].rename(columns={label_col: "label", text_col: "text"})


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Schema + label validation. Returns a cleaned frame with columns
    [label:int, text:str]. Raises DatasetError on unrecoverable problems."""
    if list(df.columns[:2]) != ["label", "text"]:
        raise DatasetError(f"Expected columns [label, text], got {list(df.columns)}")

    df = df.copy()
    df["label"] = df["label"].astype("string").str.strip().str.lower()
    df["text"] = df["text"].astype("string")

    # Missing labels / empty texts are dropped (counted in stats, not silent)
    df = df.dropna(subset=["label"])
    empty_mask = df["text"].isna() | (df["text"].str.strip() == "")
    n_empty = int(empty_mask.sum())
    if n_empty:
        logger.warning("Dropping %d rows with missing/empty text", n_empty)
    df = df.loc[~empty_mask]

    unknown = set(df["label"].unique()) - set(LABEL_MAP)
    if unknown:
        raise DatasetError(f"Unexpected label values: {sorted(unknown)}")
    df["label"] = df["label"].map(LABEL_MAP)
    # labels that parsed but are present yet unmappable (e.g. NaN) are dropped
    n_bad_label = int(df["label"].isna().sum())
    if n_bad_label:
        logger.warning("Dropping %d rows with unparsable labels", n_bad_label)
        df = df.loc[df["label"].notna()]
    df["label"] = df["label"].astype("int64")

    df = df.reset_index(drop=True)
    if len(df) == 0:
        raise DatasetError("Dataset contains no usable rows")
    return df


def dedupe_dataset(df: pd.DataFrame, inspect_n: int = 5) -> tuple[pd.DataFrame, int]:
    """Remove exact (label, text) duplicates after reporting them.

    Duplicates that straddle a train/test split leak information, so the
    deduplicated frame is what gets split.
    """
    dup_mask = df.duplicated(subset=["label", "text"], keep="first")
    n_dup = int(dup_mask.sum())
    if n_dup and logger.isEnabledFor(logging.INFO):
        examples = df.loc[dup_mask].head(inspect_n).to_dict("records")
        logger.info("Inspecting %d duplicates before removal, e.g. %s", n_dup, examples)
    return df.loc[~dup_mask].reset_index(drop=True), n_dup


def compute_stats(df: pd.DataFrame, n_duplicates: int = 0) -> dict:
    """EDA statistics dictionary, JSON-serialisable, saved for the dashboard."""
    texts = df["text"].astype(str)
    lengths = texts.str.len().to_numpy()
    word_counts = texts.str.split().map(len).to_numpy()
    counts = df["label"].value_counts().reindex(range(len(CLASS_NAMES)), fill_value=0)

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = (df["label"] == i).to_numpy()
        cls_len = lengths[mask]
        per_class[name] = {
            "count": int(mask.sum()),
            "share": float(mask.mean()),
            "mean_chars": float(cls_len.mean()) if len(cls_len) else 0.0,
            "median_chars": float(np.median(cls_len)) if len(cls_len) else 0.0,
            "min_chars": int(cls_len.min()) if len(cls_len) else 0,
            "max_chars": int(cls_len.max()) if len(cls_len) else 0,
        }

    stats = {
        "task": "sms_spam_detection",
        "n_samples": int(len(df)),
        "n_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "class_counts": {CLASS_NAMES[i]: int(counts.iloc[i]) for i in range(len(CLASS_NAMES))},
        "class_balance": {CLASS_NAMES[i]: float(counts.iloc[i] / len(df)) for i in range(len(CLASS_NAMES))},
        "imbalance_ratio": float(counts.max() / max(1, counts.min())),
        "n_duplicates_removed": int(n_duplicates),
        "n_empty_text_dropped": "see acquisition log",
        "text_length_chars": {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
            "median": float(np.median(lengths)),
        },
        "text_length_words": {
            "min": int(word_counts.min()),
            "max": int(word_counts.max()),
            "mean": float(word_counts.mean()),
            "median": float(np.median(word_counts)),
        },
        "per_class": per_class,
        "source": {
            "name": "UCI SMS Spam Collection",
            "url": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
            "license": "Provided for general and educational use (UCI terms of access)",
            "notes": "Labels mapped ham=0, spam=1; exact duplicates removed; "
                     "empty/missing texts dropped.",
        },
    }
    return stats


def top_tokens(df: pd.DataFrame, k: int = 12) -> dict:
    """Distinctive tokens per class for the Dataset Explorer (caveated)."""
    from sklearn.feature_extraction.text import CountVectorizer

    out = {}
    for i, name in enumerate(CLASS_NAMES):
        sub = df.loc[df["label"] == i, "text"].astype(str)
        if len(sub) < 10:
            out[name] = []
            continue
        vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5)
        X = vec.fit_transform(sub)
        totals = np.asarray(X.sum(axis=0)).ravel()
        order = np.argsort(-totals)[:k]
        out[name] = [
            {"term": vec.get_feature_names_out()[j], "count": int(totals[j])} for j in order
        ]
    return out


def save_stats(stats: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Dataset stats saved to %s", path)
