"""Text cleaning and preprocessing utilities.

Design notes
------------
- Preprocessing is intentionally light-touch: aggressive stemming and
  stopword removal destroys signal for short messages (SMS). Heavy work
  (lowercasing, tokenisation, n-grams) is delegated to TfidfVectorizer so
  the *same* fitted object serves training and inference.
- `TextPreprocessor` is a scikit-learn compatible transformer, so it lives
  INSIDE the Pipeline and is saved with the model. There is no separate
  inference-time cleaner that can drift from the training-time one.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_unicode(text: str) -> str:
    """NFKC normalisation keeps meaningful unicode (e.g. currency symbols
    collapsed to a canonical form) while removing compatibility oddities."""
    return unicodedata.normalize("NFKC", text)


_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_NUMBER_RE = re.compile(r"(\+?\d[\d\-\s\(\)]{6,}\d|\b\d{7,}\b)")
_REPEAT_PUNCT_RE = re.compile(r"([!?.:,;])\1+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Configurable, sklearn-pipeline-compatible text cleaner.

    Parameters
    ----------
    lower : bool
        Lowercase the text.
    strip_urls_emails : bool
        Replace URLs / e-mail addresses with placeholder tokens (they carry
        little generalisable signal and explode the vocabulary).
    normalize_numbers : bool
        Replace long digit runs (phone numbers) with a placeholder.
    collapse_punct : bool
        Collapse repeated punctuation ("!!!" -> "!").
    """

    def __init__(
        self,
        lower: bool = True,
        strip_urls_emails: bool = True,
        normalize_numbers: bool = True,
        collapse_punct: bool = True,
    ) -> None:
        self.lower = lower
        self.strip_urls_emails = strip_urls_emails
        self.normalize_numbers = normalize_numbers
        self.collapse_punct = collapse_punct

    def clean_one(self, text: object) -> str:
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return ""
        if not isinstance(text, str):
            text = str(text)
        text = normalize_unicode(text)
        if self.lower:
            text = text.lower()
        if self.strip_urls_emails:
            text = _EMAIL_RE.sub(" <email> ", text)
            text = _URL_RE.sub(" <url> ", text)
        if self.normalize_numbers:
            text = _NUMBER_RE.sub(" <number> ", text)
        if self.collapse_punct:
            text = _REPEAT_PUNCT_RE.sub(r"\1", text)
        return normalize_whitespace(text)

    def fit(self, X: Iterable[object], y=None):  # noqa: N803 - sklearn API
        # Stateless transform; fit is a no-op kept for pipeline compatibility.
        return self

    def transform(self, X: Iterable[object], y=None):  # noqa: N803
        return np.array([self.clean_one(x) for x in X], dtype=object)
