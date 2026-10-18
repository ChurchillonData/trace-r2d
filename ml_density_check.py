"""
Density check on MovieLens-20M: does retrieval stay the bottleneck on a
catalogue much denser than the Amazon one?

This is deliberately not a replication. The semantic signal is excluded,
because MovieLens item text (title plus genres, six tokens median) cannot
support a sentence encoder, and cold-start cannot be examined at all because
every MovieLens user has at least 20 ratings. The question here is narrower
and answerable: with the same two interaction-based signals, the same
budget-union combination and the same candidate-set size of 200, what recall
does retrieval reach on a denser catalogue? The 40,000-user subsample used in
the paper is 43 times denser; the measured ratio is written to the output.

Protocol matches the main paper where it can. Per-user 80/20 split, all
statistics from the training partition only, SVD with d=50, item-to-item
cosine over training co-ratings. The 33 slots the semantic source holds in the
main system are given to popularity fill, and that substitution is reported.

Writes: reports/ml_density_check.json
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

ML = Path("MovieLens/rating.csv")
OUT = Path("reports/ml_density_check.json")

SEED = 42
N_USERS = 40_000        # user subsample, each keeps all their ratings
THRESHOLD = 4.0         # MovieLens convention; 5.0 covers only 14.5%
BUDGET = 200
SLOT_SVD, SLOT_II, SLOT_POP = 100, 67, 33
N_FACTORS = 50
N_NEIGHBOURS = 50


def _load_and_core(rng: np.random.Generator) -> pd.DataFrame:
    df = pd.read_csv(ML, usecols=["userId", "movieId", "rating"])
    users = df["userId"].unique()
    keep = set(rng.choice(users, size=min(N_USERS, len(users)), replace=False))
    df = df[df["userId"].isin(keep)]
    while True:
        uc, ic = df.groupby("userId").size(), df.groupby("movieId").size()
        bad_u, bad_i = set(uc[uc < 5].index), set(ic[ic < 5].index)
        if not bad_u and not bad_i:
            break
        df = df[~df["userId"].isin(bad_u) & ~df["movieId"].isin(bad_i)]
    df["u"] = pd.factorize(df["userId"])[0]
    df["i"] = pd.factorize(df["movieId"])[0]
    return df.reset_index(drop=True)


def _split(df: pd.DataFrame, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Per-user 80/20 split, matching the main paper."""
    is_test = np.zeros(len(df), dtype=bool)
    for _, idx in df.groupby("u").indices.items():
        n_test = max(1, int(round(len(idx) * 0.2)))
        is_test[rng.choice(idx, size=n_test, replace=False)] = True
    return ~is_test, is_test


def _item_neighbours(train: csr_matrix) -> dict[int, np.ndarray]:
    """Top-N cosine neighbours per item from training co-ratings."""
    cols = normalize(train.T.tocsr(), norm="l2", axis=1)
    n_items = cols.shape[0]
    out: dict[int, np.ndarray] = {}
    block = 512
    for s in range(0, n_items, block):
        sim = (cols[s:s + block] @ cols.T).toarray()
        for r in range(sim.shape[0]):
            sim[r, s + r] = -1.0          # drop self
        k = min(N_NEIGHBOURS, sim.shape[1] - 1)
        top = np.argpartition(-sim, k, axis=1)[:, :k]
        for r in range(sim.shape[0]):
            order = top[r][np.argsort(-sim[r, top[r]])]
            out[s + r] = order[sim[r, order] > 0]
    return out


def _retrieve(u: int, liked: np.ndarray, seen: set, uf: np.ndarray,
              itf: np.ndarray, neigh: dict, pop: np.ndarray) -> set:
    """Budget-union retrieval: SVD slots, item-item slots, popularity fill."""
    pool: list[int] = []
    taken: set = set()

    scores = itf @ uf[u]
    for it in np.argsort(-scores):
        if len(taken) >= SLOT_SVD:
            break
        if it not in seen and it not in taken:
            pool.append(int(it)); taken.add(int(it))

    ii_target = len(taken) + SLOT_II
    for src in liked:
        if len(taken) >= ii_target:
            break
        for nb in neigh.get(int(src), ()):
            if len(taken) >= ii_target:
                break
            if nb not in seen and nb not in taken:
                pool.append(int(nb)); taken.add(int(nb))

    for it in pop:
        if len(taken) >= BUDGET:
            break
        if it not in seen and it not in taken:
            pool.append(int(it)); taken.add(int(it))
    return taken


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print("[ml] loading and filtering …")
    df = _load_and_core(rng)
    n_u, n_i = df["u"].nunique(), df["i"].nunique()
    density = len(df) / (n_u * n_i) * 100
    print(f"  {len(df):,} ratings  {n_u:,} users  {n_i:,} items  "
          f"density {density:.4f}%")

    tr_mask, te_mask = _split(df, rng)
    tr, te = df[tr_mask], df[te_mask]
    print(f"  train {len(tr):,}  test {len(te):,}")

    train = csr_matrix((tr["rating"].values, (tr["u"].values, tr["i"].values)),
                       shape=(n_u, n_i))

    print("[ml] SVD …")
    svd = TruncatedSVD(n_components=N_FACTORS, random_state=SEED)
    uf = svd.fit_transform(train).astype(np.float32)
    itf = svd.components_.T.astype(np.float32)

    print("[ml] item-item neighbours …")
    neigh = _item_neighbours(train)

    pop = np.array(tr["i"].value_counts().index, dtype=np.int64)
    seen_by_u = tr.groupby("u")["i"].apply(set).to_dict()
    liked_by_u = (tr[tr["rating"] >= THRESHOLD].groupby("u")["i"]
                  .apply(lambda s: s.values[:20]).to_dict())
    pos_by_u = te[te["rating"] >= THRESHOLD].groupby("u")["i"].apply(list).to_dict()

    print(f"[ml] retrieval for {len(pos_by_u):,} test users …")
    hit = tot = 0
    for n, (u, positives) in enumerate(pos_by_u.items(), 1):
        pool = _retrieve(u, liked_by_u.get(u, np.array([], dtype=np.int64)),
                         seen_by_u.get(u, set()), uf, itf, neigh, pop)
        for p in positives:
            tot += 1
            if p in pool:
                hit += 1
        if n % 5000 == 0:
            print(f"    {n:,} users   running recall {hit/tot*100:.2f}%")

    recall = hit / tot * 100
    res = {
        "dataset": "MovieLens-20M",
        "user_subsample": N_USERS,
        "ratings": int(len(df)), "users": int(n_u), "items": int(n_i),
        "density_pct": round(density, 4),
        "relevance_threshold": THRESHOLD,
        "candidate_set_size": BUDGET,
        "slots": {"svd": SLOT_SVD, "itemitem": SLOT_II, "popularity": SLOT_POP},
        "semantic_excluded": True,
        "test_positives": int(tot),
        "recall_at_200_pct": round(recall, 2),
        "amazon_comparison": {"density_pct": 0.0234, "recall_at_200_pct": 3.30},
        "density_ratio": round(density / 0.0234, 1),
        "recall_ratio": round(recall / 3.30, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"MovieLens density   {density:.4f}%   vs Amazon 0.0234%   "
          f"({res['density_ratio']}x)")
    print(f"Recall@200          {recall:.2f}%     vs Amazon 3.30%      "
          f"({res['recall_ratio']}x)")
    print("=" * 60)
    print(f"\n[ml] done in {(time.time()-t0)/60:.1f} min  ->  {OUT}")


if __name__ == "__main__":
    main()
