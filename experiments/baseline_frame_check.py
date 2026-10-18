"""
Diagnostic: how much does the evaluation denominator change the baseline story?

metrics.json reports baselines under a retrieved-only denominator, which skips
any positive that retrieval missed. The two-stage headline numbers count those
misses as failures. This script scores the same fixed pool both ways so the
two frames can be compared directly.

Writes: reports/baseline_frame_check.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import yaml

from recsys.evaluation import _hit_ndcg_mrr

POOL = Path("ranking/candidate_scores_budget200_cand.parquet")
SIGNALS = ["score", "item_popularity", "svd_score", "semantic_score", "itemitem_score"]
OUT = Path("reports/baseline_frame_check.json")
USER_BATCH = 5_000


def _positives(cfg: dict) -> dict[int, list[int]]:
    split = json.loads(Path(cfg["paths"]["split"]).read_text())
    df = pd.read_parquet(cfg["paths"]["clean"], columns=["user", "item", "rating"])
    thr = float(cfg["evaluation"]["relevance_threshold"])
    test = df.loc[split["test_idx"]]
    return {int(u): g["item"].tolist()
            for u, g in test[test["rating"] >= thr].groupby("user")}


def _eval_signal(pool: pd.DataFrame, col: str, user_pos: dict,
                 cfg: dict) -> dict:
    """Return metrics under both denominators for one ranking signal."""
    k_vals = cfg["evaluation"]["k_values"]
    neg = cfg["evaluation"]["negatives_per_positive"]
    pool_size = cfg["retrieval"]["candidate_set_size"]
    rng = np.random.default_rng(cfg["seed"])
    retrieved, allusers = [], []
    for user, g in pool.groupby("user"):
        positives = user_pos.get(int(user))
        if not positives:
            continue
        items = g["item"].values
        negs_all = g[~g["item"].isin(set(positives))]["item"].tolist()
        for pos in positives:
            if int(pos) not in items:
                # Missed by retrieval: counted only in the all-users frame.
                allusers.append(_hit_ndcg_mrr(pool_size + 1, k_vals))
                continue
            n = min(neg, len(negs_all))
            if n == 0:
                continue
            sampled = set(rng.choice(negs_all, size=n, replace=False).tolist())
            sub = g[g["item"].isin({int(pos)} | sampled)].sort_values(
                col, ascending=False)
            rank = {it: r + 1 for r, it in enumerate(sub["item"])}.get(int(pos), n + 1)
            m = _hit_ndcg_mrr(rank, k_vals)
            retrieved.append(m)
            allusers.append(m)

    def agg(rows: list) -> dict:
        if not rows:
            return {}
        d = pd.DataFrame(rows)
        out = {"n": len(d)}
        for k in k_vals:
            out[f"hr@{k}"] = round(float(d[f"hr@{k}"].mean()), 4)
            out[f"ndcg@{k}"] = round(float(d[f"ndcg@{k}"].mean()), 4)
        return out

    return {"retrieved_only": agg(retrieved), "all_users": agg(allusers)}


def main() -> None:
    cfg = yaml.safe_load(open("configs/default.yaml"))
    user_pos = _positives(cfg)
    print(f"[frame] loading {POOL.name} …")
    pool = pd.read_parquet(POOL, columns=["user", "item"] + SIGNALS)

    results = {}
    for col in SIGNALS:
        print(f"[frame] {col} …", end=" ", flush=True)
        r = _eval_signal(pool, col, user_pos, cfg)
        results[col] = r
        ro, au = r["retrieved_only"], r["all_users"]
        print(f"retrieved-only HR@10={ro.get('hr@10',0):.4f} (n={ro.get('n',0):,})   "
              f"all-users HR@10={au.get('hr@10',0):.4f} (n={au.get('n',0):,})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[frame] done  →  {OUT}")


if __name__ == "__main__":
    main()
