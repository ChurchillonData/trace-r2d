"""
Baselines: four ranked lists for comparison against the full hybrid ranker.

Each baseline reads the scored candidate list from the ranking stage and
re-sorts by a single signal. The evaluation module scores all four with
the same sampled-negative protocol it uses for the main model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _load_scored(cfg: dict) -> pd.DataFrame:
    """Load the full scored candidate list produced by the ranking stage."""
    return pd.read_parquet(Path(cfg["paths"]["ranking_dir"]) / "candidate_scores.parquet")


def _top_n(df: pd.DataFrame, score_col: str, cfg: dict) -> pd.DataFrame:
    """Sort by score_col descending per user and keep top_n rows."""
    top_n = cfg["ranking"]["top_n"]
    return (
        df.sort_values(["user", score_col], ascending=[True, False])
        .groupby("user")
        .head(top_n)
        .reset_index(drop=True)
    )


def popularity_baseline(cfg: dict) -> pd.DataFrame:
    """Rank candidates by training-set item popularity alone."""
    df = _load_scored(cfg)
    return _top_n(df.assign(score=df["item_popularity"]), "score", cfg)


def svd_baseline(cfg: dict) -> pd.DataFrame:
    """Rank candidates by SVD dot-product score alone."""
    df = _load_scored(cfg)
    return _top_n(df.assign(score=df["svd_score"]), "score", cfg)


def semantic_baseline(cfg: dict) -> pd.DataFrame:
    """Rank candidates by SBERT cosine similarity alone."""
    df = _load_scored(cfg)
    return _top_n(df.assign(score=df["semantic_score"]), "score", cfg)


def itemitem_baseline(cfg: dict) -> pd.DataFrame:
    """Rank candidates by item-to-item collaborative score alone."""
    df = _load_scored(cfg)
    return _top_n(df.assign(score=df["itemitem_score"]), "score", cfg)


def tfidf_baseline(cfg: dict) -> pd.DataFrame:
    """Rank candidates by TF-IDF text similarity score alone.

    This is the rejected baseline from the original notebook.  Its scores
    are lower than the Sentence-BERT semantic baseline, which motivates the
    upgrade to dense embeddings.
    """
    df = _load_scored(cfg)
    return _top_n(df.assign(score=df["tfidf_score"]), "score", cfg)


def build(cfg: dict) -> None:
    """Write all baseline ranked lists to ranking/."""
    rank_dir = Path(cfg["paths"]["ranking_dir"])
    names = {
        "baseline_popularity.parquet": popularity_baseline,
        "baseline_svd.parquet": svd_baseline,
        "baseline_semantic.parquet": semantic_baseline,
        "baseline_itemitem.parquet": itemitem_baseline,
        "baseline_tfidf.parquet": tfidf_baseline,
    }
    for filename, fn in names.items():
        out = rank_dir / filename
        if out.exists():
            print(f"[baselines] {filename} found, skipping.")
            continue
        df = fn(cfg)
        df.to_parquet(out, index=False)
        print(f"[baselines] {filename} written  ({df['user'].nunique():,} users)")
    print("[baselines] done.")
