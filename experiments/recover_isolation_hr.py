"""
Recover HR@1 and HR@5 for the candidate-trained ranker in isolation.

The original build_candidate_ranker.py computed all k-values (k=[1,5,10])
but only saved HR@10, NDCG@10, MRR to candidate_ranker_results.json.
This script re-runs the same isolation evaluation with model_candidate.json
and reports all five metrics. No retraining. No modification to main results.

Sanity check: HR@10 must equal 0.5167 (within rounding to 4dp).
Output: reports/isolation_hr_recovery.json  (supplementary; does not overwrite
        candidate_ranker_results.json)
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import yaml


def _load_cfg() -> dict:
    with open("configs/default.yaml") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    t0  = time.time()
    sep = "=" * 60
    print(f"\n{sep}")
    print("ISOLATION HR@1 / HR@5 RECOVERY — candidate-trained ranker")
    print(f"{sep}")

    cfg      = _load_cfg()
    paths    = cfg["paths"]
    rank_dir = Path(paths["ranking_dir"])
    feat_dir = Path(paths["features_dir"])

    cand_model = rank_dir / "model_candidate.json"
    orig_model = rank_dir / "model.json"
    bak_model  = rank_dir / "_model_orig_bak_recovery.json"

    if not cand_model.exists():
        raise FileNotFoundError(f"{cand_model} not found — candidate ranker was not trained")

    from recsys.evaluation_isolation import compute_ranker_isolation_metrics

    cold_users: set[int] = set(
        json.loads(Path(paths["cold_start"]).read_text()).get("cold_users", [])
    )

    # Temporarily swap in the candidate model, restore on exit
    shutil.copy2(orig_model, bak_model)
    shutil.copy2(cand_model, orig_model)
    print(f"[swap] model_candidate.json -> model.json (original backed up)")
    try:
        result_df = compute_ranker_isolation_metrics(cold_users, cfg, feat_dir, rank_dir)
    finally:
        shutil.copy2(bak_model, orig_model)
        bak_model.unlink(missing_ok=True)
        print(f"[swap] model.json restored")

    if result_df.empty:
        raise RuntimeError("Isolation evaluation returned empty dataframe")

    # Compute metrics for all k values
    k_values = cfg["evaluation"]["k_values"]   # [1, 5, 10]
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"hr@{k}"]   = round(float(result_df[f"hr@{k}"].mean()), 4)
        metrics[f"ndcg@{k}"] = round(float(result_df[f"ndcg@{k}"].mean()), 4)
    metrics["mrr"] = round(float(result_df["mrr"].mean()), 4)

    elapsed = (time.time() - t0) / 60
    print(f"\n[results] candidate-trained ranker, isolation (catalogue negatives):")
    print(f"  n_pairs  = {len(result_df):,}")
    print(f"  HR@1     = {metrics['hr@1']}")
    print(f"  HR@5     = {metrics['hr@5']}")
    print(f"  HR@10    = {metrics['hr@10']}")
    print(f"  NDCG@10  = {metrics['ndcg@10']}")
    print(f"  MRR      = {metrics['mrr']}")

    # Sanity check
    expected_hr10 = 0.5167
    actual_hr10   = metrics["hr@10"]
    diff          = abs(actual_hr10 - expected_hr10)
    status        = "PASS" if diff < 0.001 else "FAIL"
    print(f"\n[sanity] HR@10 expected {expected_hr10}  got {actual_hr10}  diff={diff:.4f}  -> {status}")
    if status == "FAIL":
        print("  WARNING: HR@10 does not match the stored figure. Check seed or model file.")

    print(f"\n[timing] elapsed {elapsed:.1f}m")

    # Write supplementary result file
    out_path = Path(paths["reports_dir"]) / "isolation_hr_recovery.json"
    payload = {
        "model": "candidate_ranker",
        "evaluation": "isolation (catalogue negatives)",
        "n_pairs": len(result_df),
        "seed": int(cfg["seed"]),
        "metrics": metrics,
        "sanity_hr10": {"expected": expected_hr10, "actual": actual_hr10, "status": status},
        "elapsed_minutes": round(elapsed, 1),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")
    print(sep)


if __name__ == "__main__":
    main()
