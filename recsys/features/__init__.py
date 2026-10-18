"""
Feature store: builds and assembles all per-pair score files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recsys.features import collaborative, itemitem, semantic, tfidf


def build_all(cfg: dict) -> None:
    """Run all feature stages in order."""
    feat_dir = Path(cfg["paths"]["features_dir"])
    if (feat_dir / "oof_scores.parquet").exists() and (feat_dir / "test_scores.parquet").exists():
        print("[features] assembled scores found, skipping all feature stages.")
        return
    print("[features] stage: collaborative …")
    collaborative.build(cfg)
    print("[features] stage: semantic …")
    semantic.build(cfg)
    print("[features] stage: item-to-item …")
    itemitem.build(cfg)
    print("[features] stage: tfidf (rejected baseline) …")
    tfidf.build(cfg)
    print("[features] stage: assembling scores …")
    assemble_scores(cfg)
    print("[features] all feature stages complete.")


def _build_side_features(
    df: pd.DataFrame, train_idx: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Item popularity and user activity from training data only."""
    train_df = df.loc[train_idx]
    pop = train_df["item"].value_counts().rename_axis("item").reset_index(name="item_popularity")
    act = train_df["user"].value_counts().rename_axis("user").reset_index(name="user_activity")
    return pop, act


def _merge_scores(
    base: pd.DataFrame,
    extras: list[pd.DataFrame],
    df_full: pd.DataFrame,
    item_pop: pd.DataFrame,
    user_act: pd.DataFrame,
) -> pd.DataFrame:
    """Merge score columns, add true rating and side features."""
    out = base.copy()
    for ex in extras:
        out = out.merge(ex, on=["user", "item"], how="left")
    out = out.merge(df_full[["user", "item", "rating"]], on=["user", "item"], how="left")
    out = out.merge(item_pop, on="item", how="left")
    out = out.merge(user_act, on="user", how="left")
    return out.fillna({
        "svd_score": 3.0, "semantic_score": 0.0, "itemitem_score": 0.0,
        "tfidf_score": 0.0, "item_popularity": 0, "user_activity": 0,
    })


def assemble_scores(cfg: dict) -> None:
    """Merge per-feature score files into oof_scores.parquet and test_scores.parquet."""
    paths = cfg["paths"]
    feat_dir = Path(paths["features_dir"])
    oof_out = feat_dir / "oof_scores.parquet"
    test_out = feat_dir / "test_scores.parquet"
    if oof_out.exists() and test_out.exists():
        print("[features] assembled score files found, skipping.")
        return

    df = pd.read_parquet(paths["clean"])
    split_data = json.load(open(paths["split"]))
    train_idx = split_data["train_idx"]

    svd_oof = pd.read_parquet(feat_dir / "svd_oof.parquet")
    sem_oof = pd.read_parquet(feat_dir / "semantic_oof.parquet")[["user", "item", "semantic_score"]]
    ii_oof = pd.read_parquet(feat_dir / "itemitem_oof.parquet")[["user", "item", "itemitem_score"]]
    tf_oof = pd.read_parquet(feat_dir / "tfidf_oof.parquet")[["user", "item", "tfidf_score"]]
    svd_test = pd.read_parquet(feat_dir / "svd_test.parquet")
    sem_test = pd.read_parquet(feat_dir / "semantic_test.parquet")[["user", "item", "semantic_score"]]
    ii_test = pd.read_parquet(feat_dir / "itemitem_test.parquet")[["user", "item", "itemitem_score"]]
    tf_test = pd.read_parquet(feat_dir / "tfidf_test.parquet")[["user", "item", "tfidf_score"]]

    item_pop, user_act = _build_side_features(df, train_idx)
    oof = _merge_scores(svd_oof, [sem_oof, ii_oof, tf_oof], df, item_pop, user_act)
    oof.to_parquet(oof_out, index=False)
    test = _merge_scores(svd_test, [sem_test, ii_test, tf_test], df, item_pop, user_act)
    test.to_parquet(test_out, index=False)
    print(f"[features] assembled.  oof={len(oof):,}  test={len(test):,}")
