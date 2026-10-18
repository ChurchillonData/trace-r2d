"""
Ranker-isolation evaluation using sampled negatives from the full item catalogue.

Field-standard LPO protocol: for each test user's held-out 5-star item, rank
it against negatives_per_positive unrated items sampled uniformly from the full
catalogue. Coverage is all test users with >= 1 five-star positive, independent
of whether that item was retrieved in Stage 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.evaluation import _hit_ndcg_mrr, _test_positives


def _load_iso_artifacts(features_dir: Path, ranking_dir: Path) -> dict:
    """Load SVD factors, SBERT embeddings, item-item neighbours, and ranker model."""
    import lightgbm as lgb
    factors = np.load(features_dir / "svd_factors.npz")
    neigh_df = pd.read_parquet(features_dir / "itemitem_neighbours.parquet")
    neigh_dict = {
        int(it): list(zip(g["neighbour"].tolist(), g["similarity"].tolist()))
        for it, g in neigh_df.groupby("item")
    }
    reverse_neigh: dict[int, list] = {}
    for item_id, neighs in neigh_dict.items():
        for nbr, sim in neighs:
            reverse_neigh.setdefault(int(nbr), []).append((item_id, sim))
    return {
        "u_fac": factors["user_factors"].astype(np.float32),
        "i_fac": factors["item_factors"].astype(np.float32),
        "u_embs": np.load(features_dir / "user_embeddings.npy").astype(np.float32),
        "i_embs": np.load(features_dir / "item_embeddings.npy").astype(np.float32),
        "reverse_neigh": reverse_neigh,
        "model": lgb.Booster(model_file=str(ranking_dir / "model.json")),
    }


def _training_stats(train_df: pd.DataFrame) -> tuple[dict, dict]:
    """Item popularity and user activity counts from training rows only."""
    return (
        {int(k): int(v) for k, v in train_df["item"].value_counts().items()},
        {int(k): int(v) for k, v in train_df["user"].value_counts().items()},
    )


def _ii_scores_for_user(
    candidates: list[int],
    user_train_set: set[int],
    reverse_neigh: dict[int, list],
) -> np.ndarray:
    """Item-item scores via reverse-neighbour lookup — O(|train| * k) per user."""
    accum: dict[int, float] = {}
    cnt: dict[int, int] = {}
    for j in user_train_set:
        for item_id, sim in reverse_neigh.get(j, []):
            accum[item_id] = accum.get(item_id, 0.0) + sim
            cnt[item_id] = cnt.get(item_id, 0) + 1
    return np.array(
        [accum[c] / cnt[c] if cnt.get(c, 0) > 0 else 0.0 for c in candidates],
        dtype=np.float32,
    )


def _score_batch(
    user: int, items: list[int], arts: dict,
    item_pop: dict, user_act: dict, user_train_set: set[int],
) -> np.ndarray:
    """Build ranker feature matrix for (user, items) and return model scores.

    Feature order matches FEATURE_COLS in ranking.py:
    [svd_score, semantic_score, itemitem_score, item_popularity, user_activity].
    """
    arr = np.array(items, dtype=np.int64)
    u_fac = arts["u_fac"][user]
    svd = (arts["i_fac"][arr] * u_fac).sum(axis=1).astype(np.float32)
    u_emb = arts["u_embs"][user]
    i_embs = arts["i_embs"][arr]
    u_norm = float(np.linalg.norm(u_emb))
    i_norms = np.linalg.norm(i_embs, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sem = np.where(
            (u_norm > 0) & (i_norms > 0),
            (i_embs * u_emb).sum(axis=1) / (u_norm * i_norms),
            0.0,
        ).astype(np.float32)
    ii = _ii_scores_for_user(items, user_train_set, arts["reverse_neigh"])
    pop = np.array([item_pop.get(i, 0) for i in items], dtype=np.float32)
    act = np.full(len(items), user_act.get(user, 0), dtype=np.float32)
    X = np.stack([svd, sem, ii, pop, act], axis=1)
    return arts["model"].predict(X)


def _sample_negatives(
    pos: int, all_items: np.ndarray, rated: set[int],
    n: int, rng: np.random.Generator,
) -> list[int]:
    """Sample n negatives from all_items excluding pos and all rated items."""
    exclude = rated | {pos}
    while True:
        sample = rng.choice(all_items, size=n, replace=False)
        if not any(int(s) in exclude for s in sample):
            return sample.tolist()


def _eval_iso_user(
    user: int, pos_items: list[int], all_items: np.ndarray,
    rated: set[int], arts: dict, item_pop: dict, user_act: dict,
    train_set: set[int], n_neg: int, k_values: list[int],
    rng: np.random.Generator,
) -> list[dict]:
    """Evaluate one user: each positive ranked against n_neg catalogue negatives."""
    results = []
    for pos in pos_items:
        negs = _sample_negatives(pos, all_items, rated, n_neg, rng)
        items = [pos] + negs
        scores = _score_batch(user, items, arts, item_pop, user_act, train_set)
        rank = int(np.sum(scores[1:] > scores[0])) + 1
        results.append(_hit_ndcg_mrr(rank, k_values))
    return results


def _setup_iso_eval(cfg: dict, features_dir: Path, ranking_dir: Path) -> tuple:
    """Load data, artifacts, and lookup tables for isolation evaluation."""
    df = pd.read_parquet(cfg["paths"]["clean"], columns=["user", "item", "rating"])
    split = json.load(open(cfg["paths"]["split"]))
    arts = _load_iso_artifacts(features_dir, ranking_dir)
    train_df = df.loc[split["train_idx"]]
    item_pop, user_act = _training_stats(train_df)
    user_train_sets = {int(u): set(g["item"].tolist()) for u, g in train_df.groupby("user")}
    user_rated = {int(u): set(g["item"].tolist()) for u, g in df.groupby("user")}
    all_items = np.array(sorted(df["item"].unique()), dtype=np.int64)
    threshold = float(cfg["evaluation"]["relevance_threshold"])
    user_positives = _test_positives(split, df, threshold)
    return arts, item_pop, user_act, user_train_sets, user_rated, all_items, user_positives


def compute_ranker_isolation_metrics(
    cold_users: set[int], cfg: dict,
    features_dir: Path, ranking_dir: Path,
) -> pd.DataFrame:
    """HR/NDCG/MRR for all test users with >= 1 five-star under catalogue negatives.

    Decouples ranker quality from retrieval recall. Coverage is all test users
    with at least one held-out five-star item, not just those whose item was
    retrieved into the Stage 1 candidate pool.
    """
    neg_per_pos = int(cfg["evaluation"]["negatives_per_positive"])
    k_values = cfg["evaluation"]["k_values"]
    rng = np.random.default_rng(int(cfg["seed"]))

    print("[iso-eval] loading artifacts …")
    arts, item_pop, user_act, user_train_sets, user_rated, all_items, user_positives = \
        _setup_iso_eval(cfg, features_dir, ranking_dir)
    n_total = len(user_positives)
    print(f"[iso-eval] {n_total:,} test users with >= 1 five-star positive …")

    rows = []
    for n, (user, pos_items) in enumerate(user_positives.items()):
        if n % 10_000 == 0 and n > 0:
            print(f"[iso-eval] {n:,}/{n_total:,}  ({100*n//n_total}%)")
        for m in _eval_iso_user(
            user, pos_items, all_items, user_rated.get(user, set()),
            arts, item_pop, user_act, user_train_sets.get(user, set()),
            neg_per_pos, k_values, rng,
        ):
            m["user"] = user
            m["slice"] = "cold" if user in cold_users else "warm"
            rows.append(m)

    n_pairs = len(rows)
    print(f"[iso-eval] done.  {n_total:,} users  {n_pairs:,} evaluation pairs.")
    return pd.DataFrame(rows) if rows else pd.DataFrame()
