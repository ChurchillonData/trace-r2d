"""
Oracle and near-oracle retrieval experiment.

For injection probability p in {0.0, 0.25, 0.50, 0.75, 1.0}: each test user
whose held-out positive is absent from the 200-item candidate pool has it
injected with probability p. Injection replaces the lowest-scored candidate
and uses the aligned ranker's actual score for that (user, item) pair, so the
ranker still decides the rank. p=0.0 reproduces the existing end-to-end result;
p=1.0 is the oracle (positive always present).

Writes: reports/oracle_retrieval_results.json
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from training.build_candidate_ranker import _score_candidates
from recsys.evaluation import _hit_ndcg_mrr
from recsys.ranking import FEATURE_COLS

INJECTION_PROBS = [0.0, 0.25, 0.50, 0.75, 1.0]
OUT = Path("reports/oracle_retrieval_results.json")


def _cfg() -> dict:
    with open("configs/default.yaml") as f:
        return yaml.safe_load(f)


def _load_arts(cfg: dict) -> dict:
    feat = Path(cfg["paths"]["features_dir"])
    fac = np.load(feat / "svd_factors.npz")
    split = json.loads(Path(cfg["paths"]["split"]).read_text())
    df = pd.read_parquet(cfg["paths"]["clean"], columns=["user", "item"])
    pop = df.loc[split["train_idx"]]["item"].value_counts().to_dict()
    act = df.loc[split["train_idx"]]["user"].value_counts().to_dict()
    nb = pd.read_parquet(feat / "itemitem_neighbours.parquet")
    nd = {int(it): list(zip(g["neighbour"].tolist(), g["similarity"].tolist()))
          for it, g in nb.groupby("item")}
    train_set = {int(u): set(v) for u, v in split["user_train_items"].items()}
    return {
        "u_fac": fac["user_factors"].astype(np.float32),
        "i_fac": fac["item_factors"].astype(np.float32),
        "u_embs": np.load(feat / "user_embeddings.npy").astype(np.float32),
        "item_embs": np.load(feat / "item_embeddings.npy").astype(np.float32),
        "nd": nd, "train_set": train_set, "pop": pop, "act": act,
    }


def _load_positives(cfg: dict) -> dict[int, list[int]]:
    split = json.loads(Path(cfg["paths"]["split"]).read_text())
    df = pd.read_parquet(cfg["paths"]["clean"], columns=["user", "item", "rating"])
    thr = float(cfg["evaluation"]["relevance_threshold"])
    test = df.loc[split["test_idx"]]
    return {int(u): g["item"].tolist()
            for u, g in test[test["rating"] >= thr].groupby("user")}


def _missing_pairs(
    pool_set: set[tuple[int, int]], user_pos: dict[int, list[int]]
) -> list[tuple[int, int]]:
    return [(u, p) for u, pos in user_pos.items()
            for p in pos if (u, p) not in pool_set]


def _score_missing(missing: list, arts: dict, model: lgb.Booster) -> dict[int, dict[int, float]]:
    """Return {user: {item: ranker_score}} for pairs not in the pool."""
    if not missing:
        return {}
    cands = pd.DataFrame({"user": [u for u, _ in missing],
                          "item": [i for _, i in missing]})
    feats = _score_candidates(cands, arts["u_fac"], arts["i_fac"],
                              arts["u_embs"], arts["item_embs"],
                              arts["nd"], arts["train_set"],
                              arts["pop"], arts["act"])
    feats["score"] = model.predict(feats[FEATURE_COLS].values.astype(np.float32))
    out: dict[int, dict[int, float]] = {}
    for row in feats.itertuples(index=False):
        out.setdefault(int(row.user), {})[int(row.item)] = float(row.score)
    return out


def _eval_user(user: int, pool: pd.DataFrame, positives: list[int],
               miss: dict[int, float], p: float,
               neg_per_pos: int, k_values: list[int],
               pool_size: int, rng: np.random.Generator) -> list[dict]:
    ranked = pool.copy()
    results = []
    for pos in positives:
        in_pool = int(pos) in ranked["item"].values
        if not in_pool and int(pos) in miss and rng.random() < p:
            drop_idx = ranked["score"].idxmin()
            ranked = ranked.drop(index=drop_idx)
            ranked = pd.concat(
                [ranked, pd.DataFrame({"item": [int(pos)], "score": [miss[int(pos)]]})],
                ignore_index=True)
            in_pool = True
        if not in_pool:
            rank = pool_size + 1
        else:
            negs = ranked[ranked["item"] != int(pos)]["item"].tolist()
            n = min(neg_per_pos, len(negs))
            if n == 0:
                continue
            sampled = set(rng.choice(negs, size=n, replace=False).tolist())
            sub = ranked[ranked["item"].isin({int(pos)} | sampled)].sort_values(
                "score", ascending=False)
            rank = {it: r + 1 for r, it in enumerate(sub["item"])}.get(int(pos), n + 1)
        m = _hit_ndcg_mrr(rank, k_values)
        m["user"] = user
        results.append(m)
    return results


def _eval_at_prob(pool_by_user: dict, miss_scores: dict, user_pos: dict,
                  p: float, cfg: dict) -> dict:
    k_vals = cfg["evaluation"]["k_values"]
    neg = cfg["evaluation"]["negatives_per_positive"]
    pool_size = cfg["retrieval"]["candidate_set_size"]
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for user, positives in user_pos.items():
        pool = pool_by_user.get(user)
        if pool is None:
            continue
        rows.extend(_eval_user(user, pool, positives,
                               miss_scores.get(user, {}),
                               p, neg, k_vals, pool_size, rng))
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    out: dict = {"n_users": int(df["user"].nunique())}
    for k in k_vals:
        out[f"hr@{k}"] = round(float(df[f"hr@{k}"].mean()), 4)
        out[f"ndcg@{k}"] = round(float(df[f"ndcg@{k}"].mean()), 4)
    out["mrr"] = round(float(df["mrr"].mean()), 4)
    return out


def main() -> None:
    if OUT.exists():
        print(f"[oracle] {OUT.name} exists, skip."); return
    t0 = time.time()
    cfg = _cfg()
    rank_dir = Path(cfg["paths"]["ranking_dir"])

    print("[oracle] loading artifacts …")
    arts = _load_arts(cfg)
    model = lgb.Booster(model_file=str(rank_dir / "model_candidate.json"))

    print("[oracle] loading pool …")
    pool_df = pd.read_parquet(
        rank_dir / "candidate_scores_budget200_cand.parquet",
        columns=["user", "item", "score"])
    pool_set = set(zip(pool_df["user"].tolist(), pool_df["item"].tolist()))
    pool_by_user = {int(u): df[["item", "score"]].reset_index(drop=True)
                    for u, df in pool_df.groupby("user")}

    print("[oracle] loading positives …")
    user_pos = _load_positives(cfg)

    print("[oracle] scoring missing positives …")
    missing = _missing_pairs(pool_set, user_pos)
    n_miss_users = len({u for u, _ in missing})
    print(f"  {len(missing):,} missing pairs across {n_miss_users:,} users "
          f"({n_miss_users / len(user_pos) * 100:.1f}% of users have no positive retrieved)")
    miss_scores = _score_missing(missing, arts, model)

    results = {}
    for p in INJECTION_PROBS:
        print(f"[oracle] p={p:.2f} …", end=" ", flush=True)
        m = _eval_at_prob(pool_by_user, miss_scores, user_pos, p, cfg)
        results[str(p)] = m
        print(f"HR@10={m.get('hr@10', 0):.4f}  NDCG@10={m.get('ndcg@10', 0):.4f}  "
              f"MRR={m.get('mrr', 0):.4f}  n={m.get('n_users', 0):,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"injection_probs": INJECTION_PROBS,
                               "results": results}, indent=2), encoding="utf-8")
    print(f"\n[oracle] done in {(time.time()-t0)/60:.1f} min  →  {OUT}")


if __name__ == "__main__":
    main()
