"""Pytest fixtures.

NOTE: the tiny labelled corpus here is a FABRICATED UNIT-TEST FIXTURE with
clearly synthetic patterns, deliberately separated from the real UCI dataset
used for training. It exists only so tests can verify machinery cheaply.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os  # noqa: E402

from app.core import config as app_config  # noqa: E402
from app.ml.preprocessing import TextPreprocessor  # noqa: E402
from app.ml.train import build_pipeline, classifier_specs  # noqa: E402


def pytest_configure(config):
    """Isolate test settings from the developer's .env file."""
    app_config.Settings.model_config["env_file"] = None
    app_config.get_settings.cache_clear()


@pytest.fixture(scope="session")
def tiny_dataframe() -> pd.DataFrame:
    """Synthetic toy spam/ham corpus - TEST FIXTURE ONLY, never for training."""
    spam_bits = ["free entry win cash prize", "claim your reward now", "urgent call txt stop"]
    ham_bits = ["hey meeting at noon", "see you later today", "happy birthday dear mom"]
    rows = []
    for i in range(30):
        rows.append(("spam", f"{spam_bits[i % 3]} {i} winner click"))
        rows.append(("ham", f"{ham_bits[i % 3]} {i} lunch plan"))
    return pd.DataFrame(rows, columns=["label", "text"])


@pytest.fixture(scope="session")
def tiny_csv(tmp_path_factory) -> Path:
    import numpy as np

    df = pd.DataFrame(
        [
            ("ham", "hello mum I am home soon"),
            ("spam", "WINNER! claim free prize call now"),
            ("ham", "can you bring milk tonight"),
            (np.nan, "orphan text with no label"),  # broken row: validation must drop it
            ("spam", "free free win cash urgent"),
            ("ham", "meeting moved to friday morning"),
            ("spam", "claim your reward urgent click"),
            ("ham", "I will call you later today"),
        ],
        columns=["label", "text"],
    )
    p = tmp_path_factory.mktemp("data") / "fixture.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture(scope="session")
def fitted_pipeline(tiny_dataframe):
    """A genuinely fitted (tiny) pipeline used by prediction/explain tests."""
    spec = classifier_specs()["mnb"]
    pipe = build_pipeline(spec, {"ngram_range": (1, 1), "min_df": 1, "max_features": None})
    pipe.fit(tiny_dataframe["text"], tiny_dataframe["label"].map({"ham": 0, "spam": 1}))
    return pipe


@pytest.fixture
def cleaner():
    return TextPreprocessor()
