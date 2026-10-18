"""
Stage 1 retrieval: candidate generation for test users.

For each user, unions candidates from SVD dot-product, SBERT cosine,
and item-to-item co-rating signals, removes training items, and caps
at candidate_set_size.  Falls back to popularity when history is thin.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


def _build_svd_index(item_factors: np.ndarray) -> Any:
    """FAISS dot-product index on raw SVD item factors."""
    import faiss
    index = faiss.IndexFlatIP(item_factors.shape[1])
    index.add(item_factors.astype(np.float32))
    return index


def _build_sem_index(item_embs: np.ndarray) -> tuple[np.ndarray, Any]:
    """FAISS cosine index on SBERT embeddings. Returns (normed_vecs, index)."""
    import faiss
    vecs = item_embs.astype(np.float32).copy()
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return vecs, index


def _faiss_neighbours(index: Any, query_vecs: np.ndarray, k: int) -> list[int]:
    """Query FAISS and return a flat list of neighbour item ids."""
    if len(query_vecs) == 0:
        return []
    _, ids = index.search(query_vecs.astype(np.float32), k)
    return [int(i) for row in ids for i in row if i >= 0]


def _itemitem_retrieve(
    liked: list[int],
    neigh_dict: dict[int, list[tuple[int, float]]],
    k: int,
) -> list[int]:
    """Take top-k co-rating neighbours for each liked item."""
    result = []
    for item in liked:
        result.extend(n for n, _ in neigh_dict.get(item, [])[:k])
    return result


def _take_budget(
    items: list[int], src: str, budget: int, seen: set[int],
) -> tuple[list, list]:
    """Fill up to budget; return (taken, leftover) as (item, src) pairs. Mutates seen."""
    taken, leftover = [], []
    for i in items:
        if i in seen:
            continue
        if len(taken) < budget:
            taken.append((i, src)); seen.add(i)
        else:
            leftover.append((i, src))
    return taken, leftover


def _compute_budgets(retrieval_cfg: dict, cand_size: int) -> dict[str, int]:
    """Integer slot counts from config fractions; rounding remainder goes to collaborative."""
    fracs = retrieval_cfg.get(
        "source_budgets", {"collaborative": 0.50, "itemitem": 0.335, "semantic": 0.165}
    )
    budgets = {s: int(cand_size * f) for s, f in fracs.items()}
    budgets["collaborative"] += cand_size - sum(budgets.values())
    return budgets


def _union_candidates(
    svd: list[int],
    sem: list[int],
    ii: list[int],
    pop: list[int],
    train_set: set[int],
    max_n: int,
    budgets: dict[str, int],
) -> list[tuple[int, str]]:
    """Per-source budget union; redistribute unused slots; popularity fills the rest."""
    seen = set(train_set)
    result: list[tuple[int, str]] = []
    overflow: list[tuple[int, str]] = []
    for items, src in [(svd, "collaborative"), (ii, "itemitem"), (sem, "semantic")]:
        taken, left = _take_budget(items, src, budgets.get(src, 0), seen)
        result.extend(taken)
        overflow.extend(left)
    remaining = max_n - len(result)
    for i, src in overflow:
        if remaining <= 0: break
        if i not in seen:
            result.append((i, src)); seen.add(i); remaining -= 1
    for i in pop:
        if len(result) >= max_n: break
        if i not in seen:
            result.append((i, "popularity")); seen.add(i)
    return result


def retrieve_for_user(
    user: int,
    liked: list[int],
    train_set: set[int],
    svd_index: Any,
    item_factors: np.ndarray,
    sem_index: Any,
    sem_vecs: np.ndarray,
    neigh_dict: dict[int, list[tuple[int, float]]],
    pop_items: list[int],
    cfg: dict,
) -> list[dict]:
    """Generate candidates for one user from all three signals."""
    cand_size = cfg["retrieval"]["candidate_set_size"]
    budgets = _compute_budgets(cfg["retrieval"], cand_size)
    k_per = cand_size
    if not liked:
        pairs = _union_candidates([], [], [], pop_items, train_set, cand_size, budgets)
    else:
        q = liked[:20]  # cap query size so FAISS GEMM stays bounded
        svd_c = _faiss_neighbours(svd_index, item_factors[q].astype(np.float32), k_per)
        sem_c = _faiss_neighbours(sem_index, sem_vecs[q], k_per)
        ii_c = _itemitem_retrieve(liked, neigh_dict, k_per)
        pairs = _union_candidates(svd_c, sem_c, ii_c, pop_items, train_set, cand_size, budgets)
    return [{"user": user, "item": i, "source": src} for i, src in pairs]


def _load_artifacts(cfg: dict) -> tuple:
    """Load all on-disk artifacts needed for candidate generation."""
    paths = cfg["paths"]
    feat_dir = Path(paths["features_dir"])
    df = pd.read_parquet(paths["clean"])
    split = json.load(open(paths["split"]))
    factors = np.load(feat_dir / "svd_factors.npz")
    item_embs = np.load(feat_dir / "item_embeddings.npy")
    neighbours = pd.read_parquet(feat_dir / "itemitem_neighbours.parquet")
    neigh_dict = {
        int(item): list(zip(g["neighbour"].tolist(), g["similarity"].tolist()))
        for item, g in neighbours.groupby("item")
    }
    return df, split, factors["item_factors"], item_embs, neigh_dict


def _prepare_retrieval(cfg: dict) -> tuple:
    """Load artifacts and build retrieval indices."""
    df, split, item_factors, item_embs, neigh_dict = _load_artifacts(cfg)
    threshold = cfg["retrieval"].get("rating_threshold", 4)
    train_df = df.loc[split["train_idx"]]
    liked_by_user = {int(u): g[g["rating"] >= threshold]["item"].tolist() for u, g in train_df.groupby("user")}
    pop_items = train_df["item"].value_counts().index.tolist()
    train_items_by_user = {int(u): v for u, v in split["user_train_items"].items()}
    test_users = sorted({int(u) for u in split["user_test_items"]})
    svd_index = _build_svd_index(item_factors)
    sem_vecs, sem_index = _build_sem_index(item_embs)
    return liked_by_user, pop_items, train_items_by_user, test_users, svd_index, item_factors, sem_index, sem_vecs, neigh_dict


def build(cfg: dict) -> None:
    """Run retrieval for all test users and write candidates.parquet."""
    paths = cfg["paths"]
    out_path = Path(paths["retrieval_dir"]) / "candidates.parquet"
    if out_path.exists():
        print("[retrieval] candidates found, skipping.")
        return
    print("[retrieval] loading artifacts …")
    lbu, pop, tiu, test_users, svd_i, i_fac, sem_i, sem_v, nd = _prepare_retrieval(cfg)
    print(f"[retrieval] generating candidates for {len(test_users):,} users …")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa, pyarrow.parquet as pq
    writer, total, chunk = None, 0, []
    for user in tqdm(test_users, desc="  candidates"):
        chunk.extend(retrieve_for_user(user, lbu.get(user, []), set(tiu.get(user, [])), svd_i, i_fac, sem_i, sem_v, nd, pop, cfg))
        if len(chunk) >= 1_000_000:
            tbl = pa.Table.from_pylist(chunk)
            writer = writer or pq.ParquetWriter(str(out_path), tbl.schema)
            writer.write_table(tbl); total += len(chunk); chunk = []
    if chunk:
        tbl = pa.Table.from_pylist(chunk)
        writer = writer or pq.ParquetWriter(str(out_path), tbl.schema)
        writer.write_table(tbl); total += len(chunk)
    if writer:
        writer.close()
    print(f"[retrieval] done.  {len(test_users):,} users  {total:,} candidates")
