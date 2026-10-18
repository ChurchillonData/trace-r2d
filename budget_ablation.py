"""
Extended candidate-budget ablation.

Sweeps retrieval budgets [50, 100, 150, 200, 300, 500]. For each budget:
  - Runs end-to-end retrieval (or loads cached parquet if present)
  - Scores with both the obs-pair ranker (model.json) and the aligned ranker
    (model_candidate.json)
  - Evaluates HR/NDCG/MRR and records retrieval recall

Writes: reports/budget_ablation_results.json
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from build_candidate_ranker import _score_candidates
from recsys.evaluation import _hit_ndcg_mrr
from recsys.ranking import FEATURE_COLS
from recsys.retrieval import _prepare_retrieval, retrieve_for_user

BUDGETS = [50, 100, 150, 200, 300, 500]
USER_BATCH = 5_000   # cap rows per BLAS call, avoids OOM at large budgets
OUT = Path("reports/budget_ablation_results.json")
PARTIAL = Path("reports/budget_ablation_partial.json")


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


def _retrieve_budget(budget: int, cfg: dict, retr_tuple: tuple,
                     test_users: list[int]) -> Path:
    """Retrieve for every test user and stream to parquet in chunks."""
    cache = Path(cfg["paths"]["retrieval_dir"]) / f"candidates_budget{budget}.parquet"
    if cache.exists():
        return cache
    import pyarrow as pa
    import pyarrow.parquet as pq

    cfg_b = dict(cfg)
    cfg_b["retrieval"] = dict(cfg["retrieval"], candidate_set_size=budget)
    (liked_by_user, pop_items, train_items_by_user,
     svd_index, item_factors, sem_index, sem_vecs, neigh_dict) = retr_tuple
    cache.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp path and rename on success, so a crash never leaves a
    # truncated file that the next run would mistake for a complete cache.
    tmp = cache.with_suffix(".parquet.tmp")
    writer, chunk = None, []
    try:
        for user in tqdm(test_users, desc=f"  retrieve b={budget}",
                         mininterval=5.0, miniters=2000):
            chunk.extend(retrieve_for_user(
                user, liked_by_user.get(user, []),
                set(train_items_by_user.get(user, [])),
                svd_index, item_factors, sem_index, sem_vecs,
                neigh_dict, pop_items, cfg_b))
            if len(chunk) >= 1_000_000:
                tbl = pa.Table.from_pylist(chunk)
                writer = writer or pq.ParquetWriter(str(tmp), tbl.schema)
                writer.write_table(tbl)
                chunk = []
        if chunk:
            tbl = pa.Table.from_pylist(chunk)
            writer = writer or pq.ParquetWriter(str(tmp), tbl.schema)
            writer.write_table(tbl)
    finally:
        if writer:
            writer.close()
    tmp.replace(cache)
    return cache


def _score_batch(pool: pd.DataFrame, arts: dict, model: lgb.Booster) -> pd.DataFrame:
    feats = _score_candidates(pool, arts["u_fac"], arts["i_fac"],
                              arts["u_embs"], arts["item_embs"],
                              arts["nd"], arts["train_set"],
                              arts["pop"], arts["act"])
    feats["score"] = model.predict(feats[FEATURE_COLS].values.astype(np.float32))
    return feats[["user", "item", "score"]]


def _eval_batch(scored: pd.DataFrame, user_pos: dict, k_vals: list[int],
                neg: int, pool_size: int, rng: np.random.Generator,
                rows: list, counts: list) -> None:
    for user, pool_df in scored.groupby("user"):
        positives = user_pos.get(int(user))
        if not positives:
            continue
        items = pool_df["item"].values
        for pos in positives:
            counts[1] += 1
            in_pool = int(pos) in items
            if not in_pool:
                rank = pool_size + 1
            else:
                counts[0] += 1
                negs = pool_df[pool_df["item"] != int(pos)]["item"].tolist()
                n = min(neg, len(negs))
                if n == 0:
                    continue
                sampled = set(rng.choice(negs, size=n, replace=False).tolist())
                sub = pool_df[pool_df["item"].isin({int(pos)} | sampled)].sort_values(
                    "score", ascending=False)
                rank = {it: r + 1 for r, it in enumerate(sub["item"])}.get(int(pos), n + 1)
            m = _hit_ndcg_mrr(rank, k_vals)
            m["user"] = user
            rows.append(m)


def _score_and_eval(pool_path: Path, arts: dict, model: lgb.Booster,
                    user_pos: dict, cfg: dict, seed: int,
                    pool_size: int) -> dict:
    """Stream the pool in user batches so BLAS never sees the whole matrix."""
    pool = pd.read_parquet(pool_path, columns=["user", "item"])
    users = pool["user"].unique()
    k_vals = cfg["evaluation"]["k_values"]
    neg = cfg["evaluation"]["negatives_per_positive"]
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    counts = [0, 0]   # retrieved, total
    for start in range(0, len(users), USER_BATCH):
        batch_users = set(users[start:start + USER_BATCH].tolist())
        batch = pool[pool["user"].isin(batch_users)]
        scored = _score_batch(batch, arts, model)
        _eval_batch(scored, user_pos, k_vals, neg, pool_size, rng, rows, counts)
        del scored, batch
    recall = counts[0] / counts[1] if counts[1] else 0.0
    if not rows:
        return {"recall": round(recall, 4)}
    df = pd.DataFrame(rows)
    out: dict = {"n_users": int(df["user"].nunique()),
                 "recall": round(recall, 4)}
    for k in k_vals:
        out[f"hr@{k}"] = round(float(df[f"hr@{k}"].mean()), 4)
        out[f"ndcg@{k}"] = round(float(df[f"ndcg@{k}"].mean()), 4)
    out["mrr"] = round(float(df["mrr"].mean()), 4)
    return out


def _phase_retrieve(cfg: dict) -> None:
    """Build every missing budget parquet, then free the retrieval artifacts.

    Kept separate from scoring because the two artifact sets duplicate the
    large arrays, and holding both at once exhausts memory at high budgets.
    """
    todo = [b for b in BUDGETS
            if not (Path(cfg["paths"]["retrieval_dir"]) /
                    f"candidates_budget{b}.parquet").exists()]
    if not todo:
        print("[ablation] all budget pools cached.")
        return
    print(f"[ablation] preparing retrieval artifacts for budgets {todo} …")
    (liked_by_user, pop_items, train_items_by_user, test_users,
     svd_index, item_factors, sem_index, sem_vecs, neigh_dict) = _prepare_retrieval(cfg)
    retr_tuple = (liked_by_user, pop_items, train_items_by_user,
                  svd_index, item_factors, sem_index, sem_vecs, neigh_dict)
    for budget in todo:
        print(f"[ablation] retrieving budget={budget} …")
        _retrieve_budget(budget, cfg, retr_tuple, test_users)


def _phase_score(cfg: dict) -> dict:
    """Score and evaluate every cached budget pool with both rankers."""
    rank_dir = Path(cfg["paths"]["ranking_dir"])
    print("[ablation] loading ranking artifacts …")
    arts = _load_arts(cfg)
    user_pos = _load_positives(cfg)
    model_obs = lgb.Booster(model_file=str(rank_dir / "model.json"))
    model_cand = lgb.Booster(model_file=str(rank_dir / "model_candidate.json"))

    results: dict[str, dict] = {}
    if PARTIAL.exists():
        results = json.loads(PARTIAL.read_text())["results"]
        print(f"[ablation] resuming, {len(results)} budgets already scored")

    for budget in BUDGETS:
        if str(budget) in results:
            print(f"\n[ablation] budget={budget} already scored, skip.")
            continue
        pool_path = (Path(cfg["paths"]["retrieval_dir"]) /
                     f"candidates_budget{budget}.parquet")
        if not pool_path.exists():
            print(f"\n[ablation] budget={budget} pool missing, skip.")
            continue
        print(f"\n[ablation] budget={budget}  pool={pool_path.name}")
        m_obs = _score_and_eval(pool_path, arts, model_obs, user_pos,
                                cfg, cfg["seed"], budget)
        m_cand = _score_and_eval(pool_path, arts, model_cand, user_pos,
                                 cfg, cfg["seed"], budget)
        results[str(budget)] = {"obs": m_obs, "cand": m_cand}
        print(f"  recall={m_obs['recall']:.4f}"
              f"  obs  HR@10={m_obs.get('hr@10',0):.4f}  NDCG@10={m_obs.get('ndcg@10',0):.4f}"
              f"  MRR={m_obs.get('mrr',0):.4f}")
        print(f"  recall={m_cand['recall']:.4f}"
              f"  cand HR@10={m_cand.get('hr@10',0):.4f}  NDCG@10={m_cand.get('ndcg@10',0):.4f}"
              f"  MRR={m_cand.get('mrr',0):.4f}")
        PARTIAL.write_text(json.dumps({"budgets": BUDGETS, "results": results},
                                      indent=2), encoding="utf-8")
    return results


def main() -> None:
    if OUT.exists():
        print(f"[ablation] {OUT.name} exists, skip."); return
    t0 = time.time()
    cfg = _cfg()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    _phase_retrieve(cfg)
    gc.collect()
    results = _phase_score(cfg)

    OUT.write_text(json.dumps({"budgets": BUDGETS, "results": results}, indent=2),
                   encoding="utf-8")
    PARTIAL.unlink(missing_ok=True)
    print(f"\n[ablation] done in {(time.time()-t0)/60:.1f} min  →  {OUT}")


if __name__ == "__main__":
    main()
