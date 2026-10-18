"""
TF-IDF item features: text-similarity neighbours and per-pair scores.

Included as the rejected baseline from the original notebook. Its low
similarity values (mean ~0.026) motivate the switch to Sentence-BERT.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from tqdm import tqdm


def build_tfidf_matrix(df: pd.DataFrame, train_idx: list[int]) -> tuple[np.ndarray, list[int]]:
    """Fit TF-IDF on training reviews; return (dense_normed_vecs, item_id_list)."""
    import faiss

    sub = df.loc[train_idx, ["item", "review"]]
    item_texts = sub.groupby("item")["review"].first()
    items = item_texts.index.tolist()
    vectorizer = TfidfVectorizer(max_features=50_000, min_df=2, sublinear_tf=True)
    sparse = vectorizer.fit_transform(item_texts.values)
    n_comp = max(2, min(128, sparse.shape[0] - 1, sparse.shape[1] - 1))
    print(f"  TruncatedSVD {sparse.shape[0]:,} items x {sparse.shape[1]:,} terms -> {n_comp}d ...")
    dense = TruncatedSVD(n_components=n_comp, random_state=42).fit_transform(sparse).astype(np.float32)
    faiss.normalize_L2(dense)
    return dense, items


def compute_tfidf_neighbours(df: pd.DataFrame, train_idx: list[int], top_k: int, seed: int) -> pd.DataFrame:
    """Find top-k TF-IDF-similar items per item via FAISS search."""
    import faiss

    dense, item_ids = build_tfidf_matrix(df, train_idx)
    index = faiss.IndexFlatIP(dense.shape[1])
    index.add(dense)
    sims_arr, idx_arr = index.search(dense, top_k + 1)
    id_arr = np.array(item_ids, dtype=np.int32)
    n = len(item_ids)
    rows_col = np.repeat(np.arange(n, dtype=np.int32), top_k)
    neighs_local = idx_arr[:, 1:].ravel()
    sims_col = sims_arr[:, 1:].ravel()
    valid = neighs_local >= 0
    return pd.DataFrame({
        "item": id_arr[rows_col[valid]],
        "neighbour": id_arr[neighs_local[valid]],
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
    """Average TF-IDF neighbour similarity for items in the user's training history."""
    sub = df.loc[row_idx, ["user", "item"]]
    scores = []
    for row in sub.itertuples(index=False):
        history = user_history.get(int(row.user), set())
        neighs = neigh_dict.get(int(row.item), [])
        sims = [s for n, s in neighs if n in history]
        scores.append(float(np.mean(sims)) if sims else 0.0)
    return sub.assign(tfidf_score=scores)


def oof_tfidf_scores(
    df: pd.DataFrame,
    train_idx: list[int],
    neigh_dict: dict[int, list[tuple[int, float]]],
    cfg: dict,
) -> pd.DataFrame:
    """OOF TF-IDF scores for training rows."""
    n_folds = cfg["folds"]["n_folds"]
    seed = cfg["seed"]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    arr = np.array(train_idx)
    parts: list[pd.DataFrame] = []
    for tr_pos, va_pos in tqdm(kf.split(arr), total=n_folds, desc="  tfidf folds"):
        history = _build_user_history(df, arr[tr_pos].tolist())
        parts.append(_score_pairs(df, arr[va_pos].tolist(), neigh_dict, history))
    return pd.concat(parts, ignore_index=True)


def build(cfg: dict) -> None:
    """Write TF-IDF neighbour table and score files."""
    paths = cfg["paths"]
    feat_dir = Path(paths["features_dir"])
    feat_dir.mkdir(parents=True, exist_ok=True)
    neigh_path = feat_dir / "tfidf_neighbours.parquet"
    oof_path = feat_dir / "tfidf_oof.parquet"
    test_path = feat_dir / "tfidf_test.parquet"
    if oof_path.exists() and test_path.exists() and neigh_path.exists():
        print("[tfidf] artifacts found, skipping.")
        return
    df = pd.read_parquet(paths["clean"], columns=["user", "item", "rating", "review"])
    split = json.load(open(paths["split"]))
    train_idx, test_idx = split["train_idx"], split["test_idx"]
    top_k = cfg["features"]["itemitem"]["top_neighbours"]
    print("[tfidf] building TF-IDF neighbour table …")
    neighbours = compute_tfidf_neighbours(df, train_idx, top_k, cfg["seed"])
    neighbours.to_parquet(neigh_path, index=False)
    neigh_dict: dict[int, list[tuple[int, float]]] = {
        int(item): list(zip(g["neighbour"].tolist(), g["similarity"].tolist()))
        for item, g in neighbours.groupby("item")
    }
    print("[tfidf] generating OOF scores …")
    oof = oof_tfidf_scores(df, train_idx, neigh_dict, cfg)
    oof.to_parquet(oof_path, index=False)
    print("[tfidf] scoring test partition …")
    test_history = _build_user_history(df, train_idx)
    test_df = _score_pairs(df, test_idx, neigh_dict, test_history)
    test_df.to_parquet(test_path, index=False)
    print(f"[tfidf] done.  neighbours={len(neighbours):,}  oof={len(oof):,}  test={len(test_df):,}")
