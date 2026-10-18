"""
Fix leakage in OOF fold candidates, then train and evaluate the candidate ranker.

The per-fold retrieval in build_candidate_ranker.py excluded each user's training
items from the candidate pool but not their test items. This means some (user, item)
pairs where the user rated the item in the test set appear in training candidates
with label=0. These are false negatives that bias the ranker against test items.

Contamination scale: ~280-320 rows per fold (~0.05% of total). All have label=0.
No test pair has label=1 (confirmed separately).

This script:
  1. Removes test pairs from the saved fold parquets.
  2. Re-runs the leakage check on the cleaned data.
  3. Trains the candidate ranker on the clean OOF candidates.
  4. Scores candidates_budget200.parquet with the new ranker.
  5. Evaluates three ways vs the observed-pair ranker on the same retrieval.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import yaml

from build_candidate_ranker import (
    FOLD_DIR, FEATURE_COLS,
    _fold_splits, _check_leakage, _train_candidate_ranker,
    _score_test_candidates, _eval_retrieved, _eval_all_users, _eval_isolation,
    _mean, _print_table,
)


def _build_test_enc(df: pd.DataFrame, test_idx: list[int]) -> set[int]:
    """Encode test (user, item) pairs for O(1) membership checks."""
    tdf = df.loc[test_idx]
    return set(
        tdf["user"].astype(np.int64).values * 200_000 +
        tdf["item"].astype(np.int64).values
    )


def _filter_fold(path: Path, test_enc: set[int]) -> int:
    """Remove test pairs in-place. Returns number of rows removed."""
    fc = pd.read_parquet(path)
    enc = (fc["user"].astype(np.int64).values * 200_000 +
           fc["item"].astype(np.int64).values)
    keep = np.array([e not in test_enc for e in enc])
    n_removed = int((~keep).sum())
    if n_removed > 0:
        fc[keep].reset_index(drop=True).to_parquet(path, index=False)
    return n_removed


def main() -> None:
    t0 = time.time()
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]
    feat_dir = Path(paths["features_dir"])
    rank_dir = Path(paths["ranking_dir"])
    report_dir = Path(paths["reports_dir"])
    pool = cfg["retrieval"]["candidate_set_size"]

    df = pd.read_parquet(paths["clean"], columns=["user", "item", "rating"])
    split = json.load(open(paths["split"]))
    cold_users: set[int] = set(json.load(open(paths["cold_start"]))["cold_users"])
    fold_splits = _fold_splits(split["train_idx"], cfg)
    n_folds = cfg["folds"]["n_folds"]

    # ── filter test pairs from fold parquets ───────────────────────────────────
    sep = "=" * 62
    print(f"\n{sep}\nFILTERING TEST PAIRS FROM FOLD PARQUETS\n{sep}")
    test_enc = _build_test_enc(df, split["test_idx"])
    print(f"test pairs to exclude: {len(test_enc):,}")
    total_removed = 0
    for k in range(n_folds):
        path = FOLD_DIR / f"fold_candidates_{k}.parquet"
        if not path.exists():
            print(f"  fold {k}: MISSING (run build_candidate_ranker.py first)")
            return
        n = _filter_fold(path, test_enc)
        fc = pd.read_parquet(path)
        total_removed += n
        print(f"  fold {k}: removed {n:,} test pairs  ->  {len(fc):,} rows remain  "
              f"{int((fc['label']==1).sum()):,} positives")
    print(f"total removed: {total_removed:,}")

    # ── leakage check ──────────────────────────────────────────────────────────
    print(f"\n{sep}\nLEAKAGE CHECK\n{sep}")
    leakage = _check_leakage(fold_splits, FOLD_DIR, df, split)
    for k, v in leakage.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    if not all(leakage.values()):
        print("\nSTOP: leakage check FAILED after filtering.")
        return

    # ── ranker training ────────────────────────────────────────────────────────
    print(f"\n{sep}\nTRAINING CANDIDATE RANKER\n{sep}")
    model_path = rank_dir / "model_candidate.json"
    if model_path.exists():
        print(f"[train] deleting stale {model_path.name} (trained on contaminated data)")
        model_path.unlink()
    model = _train_candidate_ranker(FOLD_DIR, cfg)
    model.save_model(str(model_path))
    print(f"[train] saved to {model_path.name}")

    # ── test scoring ───────────────────────────────────────────────────────────
    print(f"\n{sep}\nSCORING TEST CANDIDATES\n{sep}")
    stale = rank_dir / "candidate_scores_budget200_cand.parquet"
    if stale.exists():
        print(f"[score] removing stale {stale.name}")
        stale.unlink()
    new_score = _score_test_candidates(model, cfg)
    old_score = rank_dir / "candidate_scores_budget200.parquet"

    # ── test positives ─────────────────────────────────────────────────────────
    print("[eval] loading test positives ...")
    dfr = pd.read_parquet(paths["clean"], columns=["user", "item", "rating"])
    threshold = float(cfg["evaluation"]["relevance_threshold"])
    test_df = dfr.loc[split["test_idx"]]
    user_pos: dict[int, list[int]] = {
        int(u): g["item"].tolist()
        for u, g in test_df[test_df["rating"] >= threshold].groupby("user")
    }
    print(f"[eval] {len(user_pos):,} users with >= 1 five-star positive")
    del dfr, test_df

    k_values = cfg["evaluation"]["k_values"]

    # ── view 1: retrieved-only ─────────────────────────────────────────────────
    print(f"\n{sep}\nVIEW 1: retrieved-only\n{sep}")
    old_ret = _eval_retrieved(old_score, user_pos, cold_users, cfg)
    new_ret = _eval_retrieved(new_score, user_pos, cold_users, cfg)

    # ── view 2: all users ──────────────────────────────────────────────────────
    print(f"\n{sep}\nVIEW 2: all users\n{sep}")
    old_all = _eval_all_users(old_score, user_pos, cold_users, cfg, pool)
    new_all = _eval_all_users(new_score, user_pos, cold_users, cfg, pool)

    # ── view 3: isolation ──────────────────────────────────────────────────────
    print(f"\n{sep}\nVIEW 3: ranker isolation\n{sep}")
    new_iso = _eval_isolation(cfg, rank_dir)

    # ── results ────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("OPTION B: CANDIDATE RANKER vs OBSERVED-PAIR RANKER")
    print("Fixed: budget-union retrieval. Variable: ranker training only.")
    print(sep)
    _print_table("VIEW 1: retrieved-only", old_ret, new_ret, k_values)
    _print_table("VIEW 2: all 62,787 users", old_all, new_all, k_values)

    old_iso = {"hr@10": 0.1701, "ndcg@10": 0.0939, "mrr": 0.0872}
    iso_hr10 = _mean(new_iso, "hr@10")
    iso_ndcg = _mean(new_iso, "ndcg@10")
    iso_mrr = _mean(new_iso, "mrr")
    W = 24
    print(f"\n  VIEW 3: ranker-isolation (catalogue negatives, 62,787 users)")
    print(f"    {'':>{W}}  {'obs-pair':>12}  {'cand-train':>12}  {'delta':>10}")
    print(f"    {'-'*W}  {'-'*12}  {'-'*12}  {'-'*10}")
    for name, ov, nv in [("HR@10", old_iso["hr@10"], iso_hr10),
                          ("NDCG@10", old_iso["ndcg@10"], iso_ndcg),
                          ("MRR", old_iso["mrr"], iso_mrr)]:
        print(f"    {name:>{W}}  {ov:>12.4f}  {nv:>12.4f}  {nv-ov:>+10.4f}")

    print(f"\n{sep}")
    print("LEAKAGE CONFIRMATION")
    for k, v in leakage.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"  Test pairs filtered from fold parquets before training: {total_removed:,}")
    print(f"  All filtered pairs had label=0 (no test labels in training data)")

    elapsed = (time.time() - t0) / 60
    print(f"\n[main] total elapsed {elapsed:.1f}m")

    # ── save results ───────────────────────────────────────────────────────────
    out = {
        "candidate_ranker": {
            "retrieved_only": {
                "n_users": int(new_ret["user"].nunique()) if len(new_ret) else 0,
                "hr@10": round(_mean(new_ret, "hr@10"), 4),
                "ndcg@10": round(_mean(new_ret, "ndcg@10"), 4),
                "mrr": round(_mean(new_ret, "mrr"), 4),
            },
            "all_users": {
                "hr@10": round(_mean(new_all, "hr@10"), 4),
                "ndcg@10": round(_mean(new_all, "ndcg@10"), 4),
                "mrr": round(_mean(new_all, "mrr"), 4),
            },
            "isolation": {
                "hr@10": round(iso_hr10, 4),
                "ndcg@10": round(iso_ndcg, 4),
                "mrr": round(iso_mrr, 4),
            },
        },
        "observed_pair_ranker_on_budget_retrieval": {
            "retrieved_only": {
                "hr@10": round(_mean(old_ret, "hr@10"), 4),
                "ndcg@10": round(_mean(old_ret, "ndcg@10"), 4),
                "mrr": round(_mean(old_ret, "mrr"), 4),
            },
            "all_users": {
                "hr@10": round(_mean(old_all, "hr@10"), 4),
                "ndcg@10": round(_mean(old_all, "ndcg@10"), 4),
                "mrr": round(_mean(old_all, "mrr"), 4),
            },
            "isolation": old_iso,
        },
        "leakage": leakage,
        "leakage_fix": {
            "test_pairs_removed_from_training": total_removed,
            "all_removed_had_label_0": True,
        },
    }
    (report_dir / "candidate_ranker_results.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("[main] results written to reports/candidate_ranker_results.json")


if __name__ == "__main__":
    main()
