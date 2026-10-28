"""TF-IDF feature extraction configuration.

TfidfVectorizer is always embedded INSIDE the sklearn Pipeline so the
vocabulary learned on training data is the exact one used at inference —
it is never fitted on held-out test data or refitted separately.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    max_features: int | None = 50000,
    use_char_ngrams: bool = False,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """Word-level TF-IDF baseline; optionally augmented with char n-grams.

    Character n-grams are only enabled when they measurably improve the
    validation metric (see scripts/train_models.py which toggles this via
    the --use-char-ngrams flag during tuning experiments).
    """
    analyzer = "char_wb" if use_char_ngrams else "word"
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        analyzer=analyzer,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode",
    )


def tfidf_grid(use_char_ngrams: bool = False) -> dict[str, list]:
    """TF-IDF portion of the hyperparameter search space (pipeline-prefixed)."""
    grid: dict[str, list] = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2, 3],
        "tfidf__max_features": [30000, 60000],
    }
    if use_char_ngrams:
        grid["tfidf__analyzer"] = ["word", "char_wb"]
    return grid
