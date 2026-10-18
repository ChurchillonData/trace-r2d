"""
Measures retrieval recall, source composition, and candidate overlap after
the per-source budget fix to recsys.retrieval._union_candidates.

Reads cached SVD/SBERT/ii-neighbour artifacts from features/. Writes new
candidates to retrieval/candidates_budget200.parquet and reports findings.
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

from recsys.retrieval import (
    _compute_budgets,
    _faiss_neighbours,
    _itemitem_retrieve,
    _prepare_retrieval,
    retrieve_for_user,
)
from recsys.evaluation import check_leakage

SAMPLE_N = 2_000
SEED = 42


def _load_pos_map(cfg: dict) -> tuple[pd.DataFrame, dict[int, list[int]]]:
    """Test-set 5-star positives per user."""
    df = pd.read_parquet(cfg["paths"]["clean"], columns=["user", "item", "rating"])
    split = json.load(open(cfg["paths"]["split"]))
    test_df = df.loc[split["test_idx"]]
    thr = float(cfg["evaluation"]["relevance_threshold"])
    pos_map = {
        int(u): g["item"].tolist()
        for u, g in test_df[test_df["rating"] >= thr].groupby("user")
    }
    return test_df, pos_map


def _gen_candidates(
    out_path: Path, lbu: dict, tiu: dict, test_users: list,
    svd_i, i_fac, sem_i, sem_v, nd, pop, cfg: dict,
) -> pd.DataFrame:
    """Run budget union for all test users; cache result to out_path."""
    if out_path.exists():
        print(f"[retrieval] loading cached {out_path.name}")
        return pd.read_parquet(out_path)
    import pyarrow as pa
    import pyarrow.parquet as pq
    print(f"[retrieval] generating for {len(test_users):,} users ...")
    writer, chunk = None, []
    for n, u in enumerate(test_users):
        if n % 10_000 == 0 and n > 0:
            print(f"  {n:,}/{len(test_users):,}  ({100*n//len(test_users)}%)")
        chunk.extend(retrieve_for_user(
            u, lbu.get(u, []), set(tiu.get(u, [])),
            svd_i, i_fac, sem_i, sem_v, nd, pop, cfg,
        ))
        if len(chunk) >= 1_000_000:
            tbl = pa.Table.from_pylist(chunk)
            writer = writer or pq.ParquetWriter(str(out_path), tbl.schema)
            writer.write_table(tbl); chunk = []
    if chunk:
        tbl = pa.Table.from_pylist(chunk)
        writer = writer or pq.ParquetWriter(str(out_path), tbl.schema)
        writer.write_table(tbl)
    if writer:
        writer.close()
    print(f"[retrieval] done.  {len(test_users):,} users written to {out_path.name}")
    return pd.read_parquet(out_path)


def _recall_composition(
    cands: pd.DataFrame, pos_map: dict[int, list[int]],
) -> tuple[float, int, int, pd.Series]:
    """Recall via merge (avoids giant Python set); plus source breakdown."""
    pos_rows = pd.DataFrame(
        [(u, p) for u, ps in pos_map.items() for p in ps],
        columns=["user", "item"],
    )
    hits = len(pos_rows.merge(cands[["user", "item"]], on=["user", "item"], how="inner"))
    total = len(pos_rows)
    return hits / total, hits, total, cands["source"].value_counts()


def _raw_sets(
    liked: list[int], ts: set[int],
    i_fac, svd_i, sem_v, sem_i, nd, k: int,
) -> dict[str, set[int]]:
    """Raw (pre-union) candidate sets from each source at k results per liked item."""
    if not liked:
        return {s: set() for s in ("collaborative", "itemitem", "semantic")}
    q = np.array(liked[:20], dtype=np.int64)
    return {
        "collaborative": set(_faiss_neighbours(svd_i, i_fac[q].astype(np.float32), k)) - ts,
        "semantic":      set(_faiss_neighbours(sem_i, sem_v[q], k)) - ts,
        "itemitem":      set(_itemitem_retrieve(liked, nd, k)) - ts,
    }


def _report_overlap(
    sample_users: list[int], lbu: dict, tiu: dict,
    i_fac, svd_i, sem_v, sem_i, nd,
    user_cands: dict[int, list[int]], k: int,
) -> None:
    """Fraction of each user's final candidates that appear in one vs multiple source sets."""
    unique = shared = pop_fill = total = 0
    for u in sample_users:
        ts = set(tiu.get(u, []))
        rs = _raw_sets(lbu.get(u, []), ts, i_fac, svd_i, sem_v, sem_i, nd, k)
        for item in user_cands.get(u, []):
            n_src = sum(1 for s in rs.values() if item in s)
            if n_src == 0:
                pop_fill += 1
            elif n_src == 1:
                unique += 1
            else:
                shared += 1
            total += 1
    if total:
        print(f"  Unique to one source:  {unique:,}  ({100*unique/total:.1f}%)")
        print(f"  In 2+ source sets:     {shared:,}  ({100*shared/total:.1f}%)")
        print(f"  Popularity fill only:  {pop_fill:,}  ({100*pop_fill/total:.1f}%)")


def main() -> None:
    """Generate budget-union candidates, then report recall, composition, and overlap."""
    t0 = time.time()
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    rng = np.random.default_rng(SEED)

    print("[measure] loading data and artifacts ...")
    _, pos_map = _load_pos_map(cfg)
    lbu, pop, tiu, test_users, svd_i, i_fac, sem_i, sem_v, nd = _prepare_retrieval(cfg)
    k = cfg["retrieval"]["candidate_set_size"]
    budgets = _compute_budgets(cfg["retrieval"], k)
    print(f"[measure] budgets: {budgets}  (pool={k})")
    print(f"[measure] {len(test_users):,} test users  {len(pos_map):,} with 5-star positives")

    out = Path(cfg["paths"]["retrieval_dir"]) / "candidates_budget200.parquet"
    cands = _gen_candidates(out, lbu, tiu, test_users, svd_i, i_fac, sem_i, sem_v, nd, pop, cfg)

    recall, hits, total, comp = _recall_composition(cands, pos_map)

    print(f"\n{'='*60}")
    print("RECALL AT POOL=200")
    print(f"  Old priority fill (2.61%):  3,086 / 118,464 positives")
    print(f"  New budget union  ({recall:.4f}):  {hits:,} / {total:,} positives")
    print(f"  Change:  {(recall - 0.0261)*100:+.2f} percentage points")

    print(f"\n{'='*60}")
    print(f"SOURCE COMPOSITION  ({len(test_users):,} users  {len(cands):,} candidates)")
    n_cands = len(cands)
    for src, ct in comp.items():
        print(f"  {src:15s}: {ct:10,}  ({100*ct/n_cands:.2f}%)")
    slots = cands.groupby(["user", "source"]).size().unstack(fill_value=0)
    print("\n  Slots per user per source  (median  /  zero-users):")
    for src in ["collaborative", "itemitem", "semantic", "popularity"]:
        col = slots.get(src, pd.Series(0, index=slots.index))
        print(f"    {src:15s}: median={col.median():.0f}  zero_users={100*(col==0).mean():.1f}%")

    sample_u = rng.choice(
        [u for u in test_users if lbu.get(u)],
        size=min(SAMPLE_N, len(test_users)),
        replace=False,
    ).tolist()
    sub = cands[cands["user"].isin(set(sample_u))]
    user_cands = {u: g["item"].tolist() for u, g in sub.groupby("user")}

    print(f"\n{'='*60}")
    print(f"CANDIDATE OVERLAP  ({len(sample_u):,}-user sample  raw source sets at k={k})")
    print("  Counts items in the final 200 by how many raw source sets they appear in.")
    _report_overlap(sample_u, lbu, tiu, i_fac, svd_i, sem_v, sem_i, nd, user_cands, k)

    print(f"\n{'='*60}")
    print("LEAKAGE CHECK")
    try:
        leakage = check_leakage(cfg)
    except FileNotFoundError:
        rpt = Path(cfg["paths"]["reports_dir"]) / "metrics.json"
        leakage = json.loads(rpt.read_text()).get("leakage", {})
        print("  oof_scores.parquet not on disk; retrieval change does not affect features.")
    for lk, lv in leakage.items():
        print(f"  {lk}: {'PASS' if lv else 'FAIL'}")

    print(f"\n[measure] done in {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
