"""Unit tests for preprocessing, dataset validation and features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.datasets import DatasetError, compute_stats, dedupe_dataset, validate_dataset
from app.ml.features import build_tfidf
from app.ml.preprocessing import TextPreprocessor


class TestTextPreprocessor:
    def test_handles_none_and_nan(self, cleaner):
        assert cleaner.clean_one(None) == ""
        assert cleaner.clean_one(np.nan) == ""

    def test_non_string_input_coerced(self, cleaner):
        assert cleaner.clean_one(12345) != ""

    def test_whitespace_normalised(self, cleaner):
        assert cleaner.clean_one("  a \t\n  b  ") == "a b"

    def test_unicode_normalised(self, cleaner):
        # fullwidth ＡＭＡＺＩＮＧ -> AMAZING (NFKC)
        assert "amazing" in cleaner.clean_one("ＡＭＡＺＩＮＧ offer")

    def test_url_email_number_placeholders(self, cleaner):
        out = cleaner.clean_one("Visit http://bit.ly/x or mail foo@bar.com call 09061234567")
        assert "<url>" in out and "<email>" in out and "<number>" in out

    def test_punctuation_collapse(self, cleaner):
        assert cleaner.clean_one("FREE!!! now...") == "free! now."

    def test_configurable_transforms_off(self):
        p = TextPreprocessor(lower=False, strip_urls_emails=False, collapse_punct=False)
        out = p.clean_one("FREE!!! http://x.com")
        assert "FREE" in out and "http://x.com" in out

    def test_transform_consistent_batch(self, cleaner):
        arr = cleaner.transform(["Hello  World", None, "A!!!"])
        assert list(arr) == ["hello world", "", "a!"]

    def test_sklearn_clone_preserves_config(self, cleaner):
        from sklearn.base import clone

        c2 = clone(cleaner)
        assert c2.lower == cleaner.lower


class TestDatasetValidation:
    def test_valid_labels_mapped(self, tiny_csv):
        df = pd.read_csv(tiny_csv)
        out = validate_dataset(df)
        assert set(out["label"].unique()) == {0, 1}
        assert "text" in out.columns and len(out) == 7  # dropped the label-less row
        assert out["label"].dtype == "int64"

    def test_unknown_label_raises(self):
        df = pd.DataFrame([("bogus", "text one")], columns=["label", "text"])
        with pytest.raises(DatasetError):
            validate_dataset(df)

    def test_missing_columns_raise(self):
        df = pd.DataFrame([("a", "b")], columns=["x", "y"])
        with pytest.raises(DatasetError):
            validate_dataset(df)

    def test_empty_text_dropped_not_crash(self):
        df = pd.DataFrame([("ham", ""), ("ham", "real text")], columns=["label", "text"])
        out = validate_dataset(df)
        assert len(out) == 1

    def test_dedupe_removes_exact_duplicates(self):
        df = pd.DataFrame(
            [("ham", "same text"), ("ham", "same text"), ("spam", "other")],
            columns=["label", "text"],
        )
        out, n = dedupe_dataset(df)
        assert n == 1 and len(out) == 2

    def test_compute_stats_shape(self, tiny_dataframe):
        df = tiny_dataframe.copy()
        df["label"] = df["label"].map({"ham": 0, "spam": 1})
        out, _ = dedupe_dataset(df)
        stats = compute_stats(out, n_duplicates=0)
        assert stats["n_samples"] == len(out)
        assert set(stats["class_counts"]) == {"ham", "spam"}
        assert abs(sum(stats["class_balance"].values()) - 1.0) < 1e-9
        assert stats["text_length_chars"]["max"] >= stats["text_length_chars"]["min"]


class TestTfidf:
    def test_fits_only_on_given_data(self):
        vec = build_tfidf(ngram_range=(1, 2), min_df=1)
        X = vec.fit_transform(["free cash prize", "hey mum home soon"])
        assert X.shape[0] == 2
        assert X.shape[1] == vec.get_feature_names_out().size

    def test_vocab_is_train_only(self):
        vec = build_tfidf(min_df=1)
        vec.fit(["free free", "cash cash"])
        vocab = vec.vocabulary_
        assert "neverseen" not in vocab
        X = vec.transform(["neverseen free"])
        assert X.shape[1] == len(vocab)

    def test_char_analyzer_option(self):
        vec = build_tfidf(use_char_ngrams=True, min_df=1)
        X = vec.fit_transform(["abc", "abd"])
        assert X.shape[1] > 0
