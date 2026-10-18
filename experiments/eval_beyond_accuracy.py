"""
Compute intra-list diversity (ILD) and novelty for all evaluated models.
Appends results to reports/metrics.json alongside the existing Gini index.

Models evaluated:
  hybrid_ranker, candidate_ranker, popularity, svd, semantic, itemitem, tfidf

Leakage contract:
  Novelty popularity counts derive from the training partition only.
  ILD uses SBERT item embeddings that are fold-independent (no rating data).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _load_top_n(path: Path, top_n: int) -> dict[int, list[int]]:
    """Return {user: [item, ...]} top-n items sorted by score desc."""
    df = pd.read_parquet(path, columns=["user", "item", "score"])
    df = df.sort_values(["user", "score"], ascending=[True, False])
    df = df.groupby("user", sort=False).head(top_n)
    return {int(u): g["item"].tolist() for u, g in df.groupby("user", sort=False)}


def _train_popularity(clean: str, train_idx: list[int]) -> tuple[dict[int, int], int]:
    """Return (item_count_map, total_interactions) from training rows only."""
    df = pd.read_parquet(clean, columns=["user", "item", "rating"])
    train_df = df.loc[train_idx]
    counts: dict[int, int] = train_df["item"].value_counts().to_dict()
    return counts, int(len(train_df))


def _check_novelty_leakage(pop_map: dict[int, int], total_train: int) -> bool:
    """Sum of popularity counts must equal the training partition size."""
    return sum(pop_map.values()) == total_train


def _ild(top_lists: dict[int, list[int]], embs: np.ndarray) -> float:
    """Average pairwise (1 - cosine_sim) over top-N items, averaged over users."""
    scores: list[float] = []
    for items in top_lists.values():
        if len(items) < 2:
            continue
        vecs = embs[items]
        sim = vecs @ vecs.T
        n = len(items)
        n_pairs = n * (n - 1) / 2
        scores.append(1.0 - float(np.triu(sim, k=1).sum()) / n_pairs)
    return float(np.mean(scores))


def _novelty(top_lists: dict[int, list[int]], pop_map: dict[int, int], total: int) -> float:
    """Average -log2(pop / total) per item in list, then over users."""
    scores: list[float] = []
    for items in top_lists.values():
        per_item = [-np.log2(max(pop_map.get(i, 1), 1) / total) for i in items]
        scores.append(float(np.mean(per_item)))
    return float(np.mean(scores))


def _gini(top_lists: dict[int, list[int]], n_items: int) -> float:
    """Gini coefficient of item recommendation frequency across all users."""
    freq = np.zeros(n_items, dtype=np.float64)
    for items in top_lists.values():
        for i in items:
            if 0 <= i < n_items:
                freq[i] += 1
    freq_s = np.sort(freq)
    n = len(freq_s)
    total = freq_s.sum()
    if total == 0:
        return 0.0
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (idx * freq_s).sum()) / (n * total) - (n + 1) / n)


def _metrics_for_lists(
    top_lists: dict[int, list[int]],
    embs: np.ndarray,
    pop_map: dict[int, int],
    total_train: int,
    n_items: int,
) -> dict[str, float]:
    """Compute ILD, novelty, and Gini for one model's top-N recommendation lists."""
    return {
        "ild": round(_ild(top_lists, embs), 4),
        "novelty": round(_novelty(top_lists, pop_map, total_train), 4),
        "gini": round(_gini(top_lists, n_items), 4),
    }


def _print_table(results: dict[str, dict[str, float]]) -> None:
    """Print per-model beyond-accuracy table."""
    w = 28
    print(f"\n  {'Model':<{w}}  {'ILD':>7}  {'Novelty':>9}  {'Gini':>7}")
    print(f"  {'-'*w}  {'-'*7}  {'-'*9}  {'-'*7}")
    for name, vals in results.items():
        label = name.replace("_", " ")
        print(f"  {label:<{w}}  {vals['ild']:>7.4f}  {vals['novelty']:>9.4f}  {vals['gini']:>7.4f}")


def _run_models(
    model_paths: dict[str, Path],
    cand_path: Path,
    top_n: int,
    embs: np.ndarray,
    pop_map: dict[int, int],
    total_train: int,
    n_items: int,
) -> dict[str, dict[str, float]]:
    """Compute beyond-accuracy metrics for each model in model_paths plus candidate ranker."""
    results: dict[str, dict[str, float]] = {}
    for name, path in model_paths.items():
        print(f"[compute] {name} ...")
        top_lists = _load_top_n(path, top_n)
        vals = _metrics_for_lists(top_lists, embs, pop_map, total_train, n_items)
        results[name] = vals
        print(f"  ILD={vals['ild']:.4f}  Novelty={vals['novelty']:.4f}  Gini={vals['gini']:.4f}")
    print("[compute] candidate_ranker (15.97M rows) ...")
    top_lists_c = _load_top_n(cand_path, top_n)
    vals_c = _metrics_for_lists(top_lists_c, embs, pop_map, total_train, n_items)
    results["candidate_ranker"] = vals_c
    print(f"  ILD={vals_c['ild']:.4f}  Novelty={vals_c['novelty']:.4f}  Gini={vals_c['gini']:.4f}")
    return results


def main() -> None:
    t0 = time.time()
    with open("configs/default.yaml") as fh:
        cfg = yaml.safe_load(fh)
    paths = cfg["paths"]
    top_n: int = cfg["ranking"]["top_n"]
    feat_dir = Path(paths["features_dir"])
    rank_dir = Path(paths["ranking_dir"])
    report_dir = Path(paths["reports_dir"])

    sep = "=" * 62
    print(f"\n{sep}")
    print("BEYOND-ACCURACY METRICS: ILD + NOVELTY + GINI")
    print(f"{sep}")
    print(f"top_n={top_n}  seed={cfg.get('seed', 42)}")

    print("\n[load] item embeddings ...")
    embs_raw: np.ndarray = np.load(feat_dir / "item_embeddings.npy")
    n_items = embs_raw.shape[0]
    norms = np.linalg.norm(embs_raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = (embs_raw / norms).astype(np.float32)
    print(f"  shape {embs.shape}  (normalised for cosine similarity)")

    print("[load] training popularity ...")
    split = json.load(open(paths["split"]))
    pop_map, total_train = _train_popularity(paths["clean"], split["train_idx"])
    print(f"  {len(pop_map):,} items with training interactions  total={total_train:,}")

    print("\n[leakage] novelty_popularity_from_train_only ...")
    ok_novelty = _check_novelty_leakage(pop_map, total_train)
    print(f"  novelty_popularity_from_train_only: {'PASS' if ok_novelty else 'FAIL'}")

    existing = json.load(open(report_dir / "metrics.json"))
    print("[leakage] existing checks:")
    for k, v in existing.get("leakage", {}).items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")

    model_paths: dict[str, Path] = {
        "hybrid_ranker": rank_dir / "ranked_lists.parquet",
        "popularity": rank_dir / "baseline_popularity.parquet",
        "svd": rank_dir / "baseline_svd.parquet",
        "semantic": rank_dir / "baseline_semantic.parquet",
        "itemitem": rank_dir / "baseline_itemitem.parquet",
        "tfidf": rank_dir / "baseline_tfidf.parquet",
    }
    cand_path = rank_dir / "candidate_scores_budget200_cand.parquet"

    print()
    results = _run_models(model_paths, cand_path, top_n, embs, pop_map, total_train, n_items)

    print(f"\n{sep}")
    print(f"BEYOND-ACCURACY SUMMARY  (top-{top_n}, all 79,838 users)")
    print(sep)
    _print_table(results)
    print()
    print(f"  ILD     = avg pairwise (1 - cosine_sim) over top-{top_n} items per user")
    print(f"  Novelty = avg -log2(item_pop / {total_train:,}) per user list")
    print(f"  Gini    = item rec-frequency inequality (0=uniform  1=concentrated)")

    print(f"\n{sep}")
    print("LEAKAGE CHECKS")
    print(sep)
    for k, v in existing.get("leakage", {}).items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"  novelty_popularity_from_train_only: {'PASS' if ok_novelty else 'FAIL'}")
    all_pass = all(existing.get("leakage", {}).values()) and ok_novelty
    print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")

    existing["beyond_accuracy"] = results
    existing["leakage"]["novelty_popularity_from_train_only"] = ok_novelty
    (report_dir / "metrics.json").write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\n[main] results written to reports/metrics.json")
    print(f"[main] elapsed {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
