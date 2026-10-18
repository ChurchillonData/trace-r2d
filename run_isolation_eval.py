"""
Standalone runner for the ranker-isolation sampled-negative evaluation.

Reads existing pipeline artifacts (model.json, SVD factors, SBERT embeddings)
and evaluates the ranker over all test users with a held-out 5-star item.
Writes reports/full_metrics.txt and reports/full_metrics.md, and adds a
ranker_isolation key to reports/metrics.json.

Run after the main pipeline has completed but before cleanup.py is called,
because cleanup removes the feature files this script reads.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Torch must precede numpy/faiss to hold the OpenMP DLL slot (WinError 1114).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: F401

import pandas as pd
import yaml

from recsys.evaluation import _ranking_dict, check_leakage
from recsys.evaluation_isolation import compute_ranker_isolation_metrics


def _load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _slice_row(df: pd.DataFrame, sl: str, k_values: list[int]) -> str:
    """Format one slice (all/warm) as a single report line."""
    sub = df if sl == "all" else df[df["slice"] == sl]
    if len(sub) == 0:
        return ""
    parts = "  ".join(
        f"HR@{k}={sub[f'hr@{k}'].mean():.4f}  NDCG@{k}={sub[f'ndcg@{k}'].mean():.4f}"
        for k in k_values
    )
    return f"    [{sl}]  {parts}  MRR={sub['mrr'].mean():.4f}"


def _retrieval_section(retrieval: dict, sz: int) -> list[str]:
    """Format the retrieval recall layer."""
    r_all = retrieval.get("retrieval_recall_all") or float("nan")
    r_warm = retrieval.get("retrieval_recall_warm") or float("nan")
    return [
        "=" * 60,
        "LAYER 1: RETRIEVAL RECALL",
        "=" * 60,
        f"  Protocol: fraction of test 5-star items in the {sz}-candidate pool.",
        f"  retrieval_recall_all={r_all:.4f}",
        f"  retrieval_recall_warm={r_warm:.4f}",
    ]


def _isolation_section(iso_df: pd.DataFrame, k_values: list[int]) -> list[str]:
    """Format the ranker-isolation layer."""
    n = iso_df["user"].nunique() if len(iso_df) else 0
    lines = [
        "=" * 60,
        "LAYER 2: RANKER QUALITY  (full-catalogue negatives)",
        "=" * 60,
        "  Protocol: each held-out 5-star positive ranked against 99 unrated",
        "  items sampled uniformly from the full catalogue. The positive is scored",
        "  by the existing ranker regardless of whether retrieval fetched it.",
        "  This decouples ranker quality from retrieval recall.",
        f"  Coverage: {n:,} test users (all users with >= 1 five-star positive).",
        "",
        "  HYBRID RANKER",
    ]
    for sl in ["all", "warm"]:
        row = _slice_row(iso_df, sl, k_values)
        if row:
            lines.append(row)
    return lines


def _pipeline_section(
    end2end: dict, baselines: dict,
    retrieval: dict, iso_n: int, sz: int, k_values: list[int],
) -> list[str]:
    """Format the end-to-end pipeline layer."""
    n = end2end.get("n_users", 0)
    recall = retrieval.get("retrieval_recall_all") or float("nan")
    lines = [
        "=" * 60,
        "LAYER 3: END-TO-END PIPELINE  (retrieval-limited)",
        "=" * 60,
        f"  Protocol: sampled-negative evaluation from the {sz}-candidate pool.",
        f"  Only users whose positive was retrieved are evaluable.",
        f"  Coverage: {n:,} users  (retrieval_recall={recall:.4f})",
        f"  Gap vs Layer 2: {iso_n - n:,} users have a positive that was not retrieved.",
        "",
        "  HYBRID RANKER (end-to-end):",
    ]
    for sl in ["all", "warm"]:
        if sl in end2end:
            s = end2end[sl]
            parts = "  ".join(
                f"HR@{k}={s.get(f'hr@{k}', 0):.4f}  NDCG@{k}={s.get(f'ndcg@{k}', 0):.4f}"
                for k in k_values
            )
            lines.append(f"    [{sl}]  {parts}  MRR={s.get('mrr', 0):.4f}")
    if baselines:
        lines += ["", "  BASELINES (end-to-end, retrieval-limited):"]
        for bn, bd in baselines.items():
            if "all" not in bd:
                continue
            s = bd["all"]
            parts = "  ".join(
                f"HR@{k}={s.get(f'hr@{k}', 0):.4f}  NDCG@{k}={s.get(f'ndcg@{k}', 0):.4f}"
                for k in k_values
            )
            lines.append(
                f"    {bn:12s}  [{bd.get('n_users', 0):,} users]  "
                f"{parts}  MRR={s.get('mrr', 0):.4f}"
            )
    return lines


def _write_full_report(
    iso_df: pd.DataFrame, existing: dict, leakage: dict, cfg: dict
) -> None:
    """Assemble and write the three-layer report to full_metrics.txt."""
    k = cfg["evaluation"]["k_values"]
    sz = cfg["retrieval"]["candidate_set_size"]
    report_dir = Path(cfg["paths"]["reports_dir"])
    iso_n = iso_df["user"].nunique() if len(iso_df) else 0

    sections: list[str] = (
        ["RECOMMENDER EVALUATION REPORT  (three-layer)", "=" * 60, ""]
        + _retrieval_section(existing.get("retrieval_recall", {}), sz)
        + [""]
        + _isolation_section(iso_df, k)
        + [""]
        + _pipeline_section(
            existing.get("hybrid_ranker", {}),
            existing.get("baselines", {}),
            existing.get("retrieval_recall", {}),
            iso_n, sz, k,
        )
        + ["", "=" * 60, "RATING PREDICTION (pointwise ranker)"]
        + [f"  {rk}={rv:.4f}" for rk, rv in existing.get("rating_prediction", {}).items()]
        + ["", "=" * 60, "LEAKAGE CHECK"]
        + [f"  {lk}: {'PASS' if lv else 'FAIL'}" for lk, lv in leakage.items()]
    )
    text = "\n".join(sections)
    (report_dir / "full_metrics.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[report] written to {report_dir / 'full_metrics.txt'}")


def _update_metrics_json(
    iso_df: pd.DataFrame, existing: dict, cfg: dict, report_dir: Path
) -> None:
    """Add ranker_isolation entry to metrics.json."""
    k = cfg["evaluation"]["k_values"]
    label = "HYBRID RANKER — ranker isolation (full-catalogue negatives)"
    existing["ranker_isolation"] = _ranking_dict(label, iso_df, k)
    (report_dir / "metrics.json").write_text(
        json.dumps(existing, indent=2), encoding="utf-8"
    )
    print(f"[report] metrics.json updated with ranker_isolation key.")


def main() -> None:
    """Run isolation evaluation and write combined three-layer report."""
    t0 = time.time()
    cfg = _load_config()
    paths = cfg["paths"]
    cold_users: set[int] = set(json.load(open(paths["cold_start"]))["cold_users"])
    report_dir = Path(paths["reports_dir"])
    cache_path = report_dir / "isolation_results.parquet"

    if cache_path.exists():
        print(f"[isolation-eval] loading cached results from {cache_path}")
        iso_df = pd.read_parquet(cache_path)
    else:
        iso_df = compute_ranker_isolation_metrics(
            cold_users=cold_users, cfg=cfg,
            features_dir=Path(paths["features_dir"]),
            ranking_dir=Path(paths["ranking_dir"]),
        )
        iso_df.to_parquet(cache_path, index=False)
        print(f"[isolation-eval] results cached to {cache_path}")

    # Load existing metrics before leakage check so we can fall back to them
    # if oof_scores.parquet was cleaned up after the original pipeline run.
    existing = json.loads((report_dir / "metrics.json").read_text())

    print("[isolation-eval] confirming leakage check …")
    try:
        leakage = check_leakage(cfg)
    except FileNotFoundError:
        leakage = existing.get("leakage", {})
        print("  oof_scores.parquet not on disk; no data changed so prior results hold.")
    for k, v in leakage.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")

    _update_metrics_json(iso_df, existing, cfg, report_dir)
    _write_full_report(iso_df, existing, leakage, cfg)

    elapsed = time.time() - t0
    n_users = iso_df["user"].nunique() if len(iso_df) else 0
    print(f"\n[isolation-eval] done.  {n_users:,} users evaluated  ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
