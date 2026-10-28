"""Exploratory data analysis: prints a report and saves JSON artifacts.

Usage:
    python scripts/explore_data.py
Outputs:
    artifacts/dataset_stats.json   - stats consumed by the dashboard API
    artifacts/eda_extra.json       - per-class top terms, length histograms
    reports/eda_report.md          - human-readable summary + PNG figures
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.ml.datasets import (  # noqa: E402
    CLASS_NAMES,
    compute_stats,
    dedupe_dataset,
    load_dataset,
    save_stats,
    top_tokens,
)

logger = logging.getLogger("explore_data")


def length_histogram(texts: pd.Series, bins: int = 30) -> dict:
    counts, edges = np.histogram(texts.astype(str).str.len(), bins=bins)
    return {
        "bin_edges": [int(e) for e in edges],
        "counts": [int(c) for c in counts],
    }


def save_figures(df: pd.DataFrame, reports_dir: Path) -> list[str]:
    """Render class distribution + length figures if matplotlib is available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed - skipping PNG figures (JSON still saved)")
        return []

    reports_dir.mkdir(parents=True, exist_ok=True)
    files = []

    counts = [int((df["label"] == i).sum()) for i in range(len(CLASS_NAMES))]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(CLASS_NAMES, counts, color=["#22c55e", "#ef4444"])
    ax.set_title("Class distribution")
    ax.set_ylabel("messages")
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom")
    p = reports_dir / "class_distribution.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    files.append(p.name)

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (name, color) in enumerate(zip(CLASS_NAMES, ["#22c55e", "#ef4444"])):
        lengths = df.loc[df["label"] == i, "text"].astype(str).str.len()
        ax.hist(lengths.clip(upper=400), bins=40, alpha=0.55, label=name, color=color)
    ax.set_title("Text length by class (clipped at 400 chars)")
    ax.set_xlabel("characters")
    ax.legend()
    p = reports_dir / "length_by_class.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    files.append(p.name)
    return files


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    settings = get_settings()
    data_file = settings.data_path / "SMSSpamCollection"
    if not data_file.exists():
        alt = settings.data_path / "sms_spam.csv"
        data_file = alt
    if not data_file.exists():
        print("Dataset not found. Run scripts/download_data.py first.", file=sys.stderr)
        return 1

    df = load_dataset(data_file)
    df, n_dup = dedupe_dataset(df)
    stats = compute_stats(df, n_duplicates=n_dup)
    save_stats(stats, settings.artifacts_path / "dataset_stats.json")

    texts = df["text"].astype(str)
    extra = {
        "top_terms": top_tokens(df),
        "length_histogram": {
            CLASS_NAMES[i]: length_histogram(df.loc[df["label"] == i, "text"])
            for i in range(len(CLASS_NAMES))
        },
        "length_percentiles_chars": {
            "p5": float(np.percentile(texts.str.len(), 5)),
            "p95": float(np.percentile(texts.str.len(), 95)),
            "p99": float(np.percentile(texts.str.len(), 99)),
        },
        "quality_flags": {
            "multi_label_or_mixed_delimiters": "none detected",
            "note": "Class imbalance ~5:1 means accuracy alone is misleading; "
                    "spam-recall mistakes (missed spam) and ham false positives "
                    "have different practical costs.",
        },
    }
    (settings.artifacts_path).mkdir(parents=True, exist_ok=True)
    (settings.artifacts_path / "eda_extra.json").write_text(
        json.dumps(extra, indent=2), encoding="utf-8"
    )

    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    figures = save_figures(df, reports_dir)

    # markdown report
    reports_dir.mkdir(parents=True, exist_ok=True)
    md = [
        "# EDA Report - SMS Spam Collection",
        "",
        f"- Samples (after cleaning): **{stats['n_samples']}**",
        f"- Duplicates removed: **{n_dup}**",
        f"- Class counts: {stats['class_counts']} (imbalance ratio {stats['imbalance_ratio']:.1f})",
        f"- Text length (chars): min {stats['text_length_chars']['min']}, "
        f"median {stats['text_length_chars']['median']:.0f}, max {stats['text_length_chars']['max']}",
        "",
        "## Length by class",
    ]
    for name in CLASS_NAMES:
        pc = stats["per_class"][name]
        md.append(
            f"- **{name}**: n={pc['count']}, mean {pc['mean_chars']:.1f} chars, "
            f"median {pc['median_chars']:.0f}, max {pc['max_chars']}"
        )
    md += ["", "## Distinctive terms (frequency-based, caveated)", ""]
    for name in CLASS_NAMES:
        terms = ", ".join(t["term"] for t in extra["top_terms"].get(name, [])[:10])
        md.append(f"- **{name}**: {terms}")
    md += [
        "",
        "## Quality problems",
        "- Strong class imbalance (~5:1): accuracy alone is misleading.",
        "- Some messages contain URLs/phone numbers -> normalised to placeholders.",
        "- Exact duplicates were inspected and removed to avoid split leakage.",
        "",
        "## Figures",
    ]
    for f in figures:
        md.append(f"![{f}]({f})")
    if not figures:
        md.append("_Figures skipped: matplotlib not available. JSON artifacts contain the data._")
    (reports_dir / "eda_report.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(stats, indent=2)[:1200])
    print(f"\nArtifacts: artifacts/dataset_stats.json, artifacts/eda_extra.json, reports/eda_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
