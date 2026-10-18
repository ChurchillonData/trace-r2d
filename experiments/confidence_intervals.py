"""
Confidence intervals via repeated seeds.

Trains the aligned ranker on the existing OOF fold candidates five times, each
with a different random seed. Scores the same held-out test pool each time and
evaluates. Reports mean and 95% confidence interval (1.96 * std) for HR, NDCG,
and MRR at each k.

Writes: reports/confidence_intervals.json
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import yaml

from training.build_candidate_ranker import _score_candidates, _train_candidate_ranker
from recsys.evaluation import _hit_ndcg_mrr
from recsys.ranking import FEATURE_COLS

SEEDS = [42, 123, 456, 789, 1000]
FOLD_DIR = Path("features/oof_fold_candidates")
OUT = Path("reports/confidence_intervals.json")


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


def _score_pool(pool_path: Path, arts: dict, model) -> pd.DataFrame:
    pool = pd.read_parquet(pool_path, columns=["user", "item"])
    feats = _score_candidates(pool, arts["u_fac"], arts["i_fac"],
                              arts["u_embs"], arts["item_embs"],
                              arts["nd"], arts["train_set"],
                              arts["pop"], arts["act"])
    feats["score"] = model.predict(feats[FEATURE_COLS].values.astype(np.float32))
    return feats[["user", "item", "score"]]


def _eval_pool(scored: pd.DataFrame, user_pos: dict, cfg: dict,
               seed: int) -> dict:
    k_vals = cfg["evaluation"]["k_values"]
    neg = cfg["evaluation"]["negatives_per_positive"]
    pool_size = cfg["retrieval"]["candidate_set_size"]
    rng = np.random.default_rng(seed)
    rows = []
    for user, pool_df in scored.groupby("user"):
        positives = user_pos.get(int(user))
        if not positives:
            continue
        for pos in positives:
            in_pool = int(pos) in pool_df["item"].values
            if not in_pool:
                rank = pool_size + 1
            else:
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
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    out: dict = {"n_users": int(df["user"].nunique())}
    for k in k_vals:
        out[f"hr@{k}"] = float(df[f"hr@{k}"].mean())
        out[f"ndcg@{k}"] = float(df[f"ndcg@{k}"].mean())
    out["mrr"] = float(df["mrr"].mean())
    return out


def _summarise(per_seed: list[dict], k_vals: list[int]) -> dict:
    metrics = [f"hr@{k}" for k in k_vals] + [f"ndcg@{k}" for k in k_vals] + ["mrr"]
    summary: dict = {}
    for m in metrics:
        vals = np.array([r[m] for r in per_seed if m in r], dtype=float)
        if len(vals) == 0:
            continue
        summary[m] = {
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std(ddof=1)), 4),
            "ci95": round(float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals))), 4),
            "min": round(float(vals.min()), 4),
            "max": round(float(vals.max()), 4),
        }
    return summary


def main() -> None:
    if OUT.exists():
        print(f"[ci] {OUT.name} exists, skip."); return
    t0 = time.time()
    cfg = _cfg()
    pool_path = Path(cfg["paths"]["ranking_dir"]) / "candidate_scores_budget200_cand.parquet"

    print("[ci] loading artifacts …")
    arts = _load_arts(cfg)
    user_pos = _load_positives(cfg)

    per_seed: list[dict] = []
    seed_rows: list[dict] = []
    for seed in SEEDS:
        print(f"[ci] seed={seed} — training ranker …", end=" ", flush=True)
        cfg_seed = dict(cfg, seed=seed)
        model = _train_candidate_ranker(FOLD_DIR, cfg_seed)
        print("scoring …", end=" ", flush=True)
        scored = _score_pool(pool_path, arts, model)
        print("evaluating …", end=" ", flush=True)
        m = _eval_pool(scored, user_pos, cfg, seed)
        per_seed.append(m)
        row: dict = {"seed": seed, **m}
        seed_rows.append(row)
        print(f"HR@10={m.get('hr@10', 0):.4f}  NDCG@10={m.get('ndcg@10', 0):.4f}  "
              f"MRR={m.get('mrr', 0):.4f}")

    summary = _summarise(per_seed, cfg["evaluation"]["k_values"])
    print("\n[ci] Summary (mean ± CI95):")
    for k in cfg["evaluation"]["k_values"]:
        for met in [f"hr@{k}", f"ndcg@{k}"]:
            s = summary.get(met, {})
            print(f"  {met}: {s.get('mean', 0):.4f} ± {s.get('ci95', 0):.4f}")
    s = summary.get("mrr", {})
    print(f"  mrr: {s.get('mean', 0):.4f} ± {s.get('ci95', 0):.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"seeds": SEEDS, "per_seed": seed_rows,
                               "summary": summary}, indent=2), encoding="utf-8")
    print(f"\n[ci] done in {(time.time()-t0)/60:.1f} min  →  {OUT}")


if __name__ == "__main__":
    main()
