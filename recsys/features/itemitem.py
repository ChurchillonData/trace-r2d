"""
Item-to-item features: co-rating similarity table and per-pair scores.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold
from tqdm import tqdm


def build_item_matrix(df: pd.DataFrame, train_idx: list[int]) -> csr_matrix:
    """Build sparse (n_items, n_users) co-rating matrix from training rows."""
    sub = df.loc[train_idx]
    n_users = int(df["user"].max()) + 1
    n_items = int(df["item"].max()) + 1
    return csr_matrix(
        (sub["rating"].astype(np.float32).values, (sub["item"].values, sub["user"].values)),
        shape=(n_items, n_users),
    )


def compute_neighbours(item_matrix: csr_matrix, top_k: int, seed: int) -> pd.DataFrame:
    """Find top-k cosine-similar items via TruncatedSVD + FAISS search."""
    import faiss
    from sklearn.decomposition import TruncatedSVD

    n_items, n_users = item_matrix.shape
    n_comp = max(2, min(64, n_users - 1, n_items - 1))
    print(f"  TruncatedSVD {n_items:,} items to {n_comp}d …")
    dense = TruncatedSVD(n_components=n_comp, random_state=seed).fit_transform(item_matrix)
    dense = dense.astype(np.float32)
    faiss.normalize_L2(dense)
    index = faiss.IndexFlatIP(n_comp)
    index.add(dense)
    sims_arr, idx_arr = index.search(dense, top_k + 1)
    items_col = np.repeat(np.arange(n_items, dtype=np.int32), top_k)
    neighs_col = idx_arr[:, 1:].ravel()
    sims_col = sims_arr[:, 1:].ravel()
    valid = neighs_col >= 0
    return pd.DataFrame({
        "item": items_col[valid],
        "neighbour": neighs_col[valid].astype(np.int32),
        "similarity": sims_col[valid].astype(np.float32),
    })


def _build_user_history(df: pd.DataFrame, row_idx: list[int]) -> dict[int, set[int]]:
    """Return {user_id: set of item_ids} from the given rows."""
    sub = df.loc[row_idx, ["user", "item"]]
    return {int(u): set(g["item"].tolist()) for u, g in sub.groupby("user")}


def _score_pairs(
    df: pd.DataFrame,
    row_idx: list[int],
    neigh_dict: dict[int, list[tuple[int, float]]],
    user_history: dict[int, set[int]],
) -> pd.DataFrame:
    """Compute itemitem_score for each (user, item) pair."""
    sub = df.loc[row_idx, ["user", "item"]]
    scores = []
    for row in sub.itertuples(index=False):
        history = user_history.get(int(row.user), set())
        neighs = neigh_dict.get(int(row.item), [])
        sims = [s for n, s in neighs if n in history]
        scores.append(float(np.mean(sims)) if sims else 0.0)
    return sub.assign(itemitem_score=scores)


def oof_itemitem_scores(
    df: pd.DataFrame,
    train_idx: list[int],
    neigh_dict: dict[int, list[tuple[int, float]]],
    cfg: dict,
) -> pd.DataFrame:
    """OOF item-item scores for training rows."""
    n_folds = cfg["folds"]["n_folds"]
    seed = cfg["seed"]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    arr = np.array(train_idx)
    parts: list[pd.DataFrame] = []
    for tr_pos, va_pos in tqdm(kf.split(arr), total=n_folds, desc="  itemitem folds"):
        history = _build_user_history(df, arr[tr_pos].tolist())
        parts.append(_score_pairs(df, arr[va_pos].tolist(), neigh_dict, history))
    return pd.concat(parts, ignore_index=True)


def build(cfg: dict) -> None:
    """Write neighbour table and item-item score files."""
    paths = cfg["paths"]
    feat_dir = Path(paths["features_dir"])
    feat_dir.mkdir(parents=True, exist_ok=True)
    neigh_path = feat_dir / "itemitem_neighbours.parquet"
    oof_path = feat_dir / "itemitem_oof.parquet"
    test_path = feat_dir / "itemitem_test.parquet"
    if oof_path.exists() and test_path.exists() and neigh_path.exists():
        print("[itemitem] artifacts found, skipping.")
        return
    df = pd.read_parquet(paths["clean"])
    split = json.load(open(paths["split"]))
    train_idx, test_idx = split["train_idx"], split["test_idx"]
    top_k = cfg["features"]["itemitem"]["top_neighbours"]
    print("[itemitem] building item-item neighbour table …")
    neighbours = compute_neighbours(build_item_matrix(df, train_idx), top_k, cfg["seed"])
    neighbours.to_parquet(neigh_path, index=False)
    neigh_dict: dict[int, list[tuple[int, float]]] = {
        int(item): list(zip(g["neighbour"].tolist(), g["similarity"].tolist()))
        for item, g in neighbours.groupby("item")
    }
    print("[itemitem] generating OOF scores …")
    oof = oof_itemitem_scores(df, train_idx, neigh_dict, cfg)
    oof.to_parquet(oof_path, index=False)
    print("[itemitem] scoring test partition …")
    test_history = _build_user_history(df, train_idx)
    test_df = _score_pairs(df, test_idx, neigh_dict, test_history)
    test_df.to_parquet(test_path, index=False)
    print(f"[itemitem] done.  neighbours={len(neighbours):,}  oof={len(oof):,}  test={len(test_df):,}")
