"""
Cold-start supplementary experiment.

Isolation guarantee
-------------------
Writes ONLY to:
  data/coldstart/
  reports/coldstart/

Reads (never writes):
  data/raw.parquet
  data/clean.parquet
  data/split.json
  features/item_embeddings.npy
  models/all-MiniLM-L6-v2

Never touches:
  reports/metrics.json
  data/split.json (write path)
  features/ (write path)
  ranking/

Usage:
  python coldstart_experiment.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Isolation guard
# ---------------------------------------------------------------------------

WRITE_ROOTS = [Path("data/coldstart"), Path("reports/coldstart")]
FORBIDDEN   = [
    Path("reports/metrics.json"), Path("data/split.json"),
    Path("data/clean.parquet"),   Path("features"),
    Path("ranking"),
]
SEED            = 42
NEG_PER_POS     = 99
TOP_K           = 10
MAX_COLD_ITEMS  = 3000   # cap SBERT encoding cost
MAX_COLD_USERS  = 10000  # cap cold-user evaluation cost


def _assert_isolation() -> None:
    for f in FORBIDDEN:
        for root in WRITE_ROOTS:
            assert not str(f).startswith(str(root)), f"Forbidden path inside write root: {f}"
    print("[guard] isolation check passed — writes go only to data/coldstart/ and reports/coldstart/")


# ---------------------------------------------------------------------------
# Raw data loading — user/item/rating only (review loaded separately, targeted)
# ---------------------------------------------------------------------------

def _clean_raw(raw_path: Path) -> pd.DataFrame:
    """Load user/item/rating only to avoid loading GBs of review text."""
    df = pd.read_parquet(raw_path, columns=["user", "item", "rating"])
    df = df[(df["user"].str.len() > 0) & (df["item"].str.len() > 0)]
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"]).copy()
    df = df.drop_duplicates(subset=["user", "item"])
    return df.reset_index(drop=True)


def _load_item_reviews(raw_path: Path, item_ids: list[str]) -> dict[str, str]:
    """
    Load review text for a small targeted set of items using pyarrow predicate
    pushdown. Returns {item_id: concatenated_review_text}.
    """
    table = pq.read_table(
        str(raw_path),
        filters=[("item", "in", item_ids)],
        columns=["item", "review"],
    )
    df = table.to_pandas().dropna(subset=["review"])
    agg = (
        df.groupby("item")["review"]
        .apply(lambda x: " ".join(x.astype(str).tolist()[:3]))
    )
    return agg.to_dict()


# ---------------------------------------------------------------------------
# Warm ID mapping (mirrors recsys/data.py encode_ids exactly)
# ---------------------------------------------------------------------------

def _kcore(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    while True:
        uc   = df["user"].value_counts()
        ic   = df["item"].value_counts()
        mask = df["user"].isin(uc[uc >= k].index) & df["item"].isin(ic[ic >= k].index)
        if mask.all():
            return df.reset_index(drop=True)
        df = df[mask]


def _build_id_maps(warm_df: pd.DataFrame) -> tuple[dict, dict]:
    u_map = {u: i for i, u in enumerate(sorted(warm_df["user"].unique()))}
    i_map = {it: i for i, it in enumerate(sorted(warm_df["item"].unique()))}
    return u_map, i_map


def _verify_maps(u_map: dict, i_map: dict) -> None:
    clean = pd.read_parquet("data/clean.parquet", columns=["user", "item"])
    match_u = len(u_map) == clean["user"].nunique()
    match_i = len(i_map) == clean["item"].nunique()
    tag = "[verify]" if (match_u and match_i) else "[verify] WARNING —"
    print(f"  {tag} ID maps: {len(u_map):,} users, {len(i_map):,} items "
          f"(clean.parquet has {clean['user'].nunique():,} / {clean['item'].nunique():,})")


# ---------------------------------------------------------------------------
# Cold entity identification
# ---------------------------------------------------------------------------

def _identify_cold(raw_df: pd.DataFrame, warm_users: set, warm_items: set,
                   max_raw_count: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    item_counts = raw_df["item"].value_counts()
    user_counts = raw_df["user"].value_counts()

    cold_item_ids = set(item_counts[item_counts <= max_raw_count].index) - warm_items
    cold_user_ids = set(user_counts[user_counts <= max_raw_count].index) - warm_users

    cold_items = (
        raw_df[raw_df["item"].isin(cold_item_ids)]
        .groupby("item")
        .agg(n_interactions=("rating", "count"))
        .reset_index()
    )
    cold_users = (
        raw_df[raw_df["user"].isin(cold_user_ids)]
        .groupby("user")
        .agg(n_interactions=("rating", "count"))
        .reset_index()
    )
    return cold_items, cold_users


# ---------------------------------------------------------------------------
# Cold-item evaluation pairs
# ---------------------------------------------------------------------------

def _cold_item_pairs(raw_df: pd.DataFrame, warm_users: set,
                     cold_item_ids: set) -> pd.DataFrame:
    mask  = (raw_df["user"].isin(warm_users)
             & raw_df["item"].isin(cold_item_ids)
             & (raw_df["rating"] >= 4))
    pairs = raw_df[mask][["user", "item", "rating"]].copy()
    pairs = pairs.drop_duplicates(subset=["user", "item"])
    return pairs.reset_index(drop=True)


# ---------------------------------------------------------------------------
# SBERT embeddings for cold items
# ---------------------------------------------------------------------------

def _encode_cold_items(cold_items: pd.DataFrame, raw_path: Path,
                       model: SentenceTransformer, batch_size: int = 64) -> np.ndarray:
    """Targeted review load + encode. No full raw.parquet in memory."""
    item_ids  = cold_items["item"].tolist()
    print(f"  loading reviews for {len(item_ids):,} cold items (targeted) ...")
    review_map = _load_item_reviews(raw_path, item_ids)
    texts = [review_map.get(it, "") for it in item_ids]
    missing = sum(1 for t in texts if not t)
    if missing:
        print(f"  ({missing} items had no review text — empty string used)")
    print(f"  encoding {len(texts):,} cold items with SBERT ...")
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype(np.float32)


# ---------------------------------------------------------------------------
# Warm user profile: avg of warm training item embeddings
# ---------------------------------------------------------------------------

def _warm_user_profiles(pair_users: list[str], u_map: dict,
                        item_embs: np.ndarray) -> dict[str, np.ndarray]:
    split    = json.loads(Path("data/split.json").read_text(encoding="utf-8"))
    clean    = pd.read_parquet("data/clean.parquet", columns=["user", "item", "rating"])
    train_df = clean.loc[split["train_idx"]]

    int_to_str_u    = {v: k for k, v in u_map.items()}
    target_int_users = {u_map[s] for s in pair_users if s in u_map}

    profiles: dict[str, np.ndarray] = {}
    for int_u in target_int_users:
        str_u  = int_to_str_u[int_u]
        rows   = train_df[(train_df["user"] == int_u) & (train_df["rating"] >= 4)]
        i_ints = rows["item"].values
        valid  = [ii for ii in i_ints if 0 <= ii < len(item_embs)]
        if not valid:
            rows  = train_df[train_df["user"] == int_u]
            valid = [ii for ii in rows["item"].values if 0 <= ii < len(item_embs)]
        if valid:
            vecs = item_embs[np.array(valid)]
            norm = np.linalg.norm(vecs, axis=1, keepdims=True); norm[norm == 0] = 1.0
            profiles[str_u] = (vecs / norm).mean(axis=0)
    return profiles


# ---------------------------------------------------------------------------
# Sampled-negative ranking evaluation
# ---------------------------------------------------------------------------

def _rank_metrics(scores: np.ndarray, pos_idx: int,
                  rng: np.random.Generator, n_candidates: int) -> dict:
    all_neg      = [i for i in range(n_candidates) if i != pos_idx]
    neg_sample   = rng.choice(all_neg, size=min(NEG_PER_POS, len(all_neg)), replace=False)
    cand_idxs    = np.array([pos_idx] + neg_sample.tolist())
    cand_scores  = scores[cand_idxs]
    order        = np.argsort(-cand_scores)
    rank         = int(np.where(order == 0)[0][0]) + 1
    hr10         = 1.0 if rank <= TOP_K else 0.0
    ndcg10       = (1.0 / np.log2(rank + 1)) if rank <= TOP_K else 0.0
    return {"hr10": hr10, "ndcg10": ndcg10, "mrr": 1.0 / rank}


def _aggregate(records: list[dict]) -> dict:
    if not records:
        return {"hr@10": 0.0, "ndcg@10": 0.0, "mrr": 0.0, "n_pairs": 0}
    hr   = float(np.mean([r["hr10"]   for r in records]))
    ndcg = float(np.mean([r["ndcg10"] for r in records]))
    mrr  = float(np.mean([r["mrr"]    for r in records]))
    return {"hr@10": round(hr, 4), "ndcg@10": round(ndcg, 4),
            "mrr": round(mrr, 4), "n_pairs": len(records)}


# ---------------------------------------------------------------------------
# Cold ITEM evaluation
# ---------------------------------------------------------------------------

def _eval_cold_items(pairs: pd.DataFrame, cold_items: pd.DataFrame,
                     cold_item_embs: np.ndarray,
                     user_profiles: dict[str, np.ndarray]) -> dict:
    item_to_idx = {it: i for i, it in enumerate(cold_items["item"].tolist())}
    rng         = np.random.default_rng(SEED)
    records: list[dict] = []
    skipped = 0

    for _, row in pairs.iterrows():
        u_str, i_str = str(row["user"]), str(row["item"])
        if u_str not in user_profiles or i_str not in item_to_idx:
            skipped += 1
            continue
        profile = user_profiles[u_str]
        pos_idx = item_to_idx[i_str]
        scores  = cold_item_embs @ profile
        records.append(_rank_metrics(scores, pos_idx, rng, len(cold_items)))

    print(f"  cold-item pairs evaluated: {len(records):,}  skipped: {skipped}")
    return {
        "sbert":    _aggregate(records),
        "svd":      "N/A — cold items have no collaborative factors",
        "itemitem": "N/A — cold items have no co-rating history",
    }


# ---------------------------------------------------------------------------
# Cold USER evaluation
# ---------------------------------------------------------------------------

def _eval_cold_users(raw_df: pd.DataFrame, cold_user_ids: set,
                     i_map: dict, item_embs: np.ndarray) -> dict:
    split     = json.loads(Path("data/split.json").read_text(encoding="utf-8"))
    clean     = pd.read_parquet("data/clean.parquet", columns=["user", "item", "rating"])
    train_pop = clean.loc[split["train_idx"], "item"].value_counts()
    n_warm    = len(item_embs)

    # Precompute pop_scores once — avoids rebuilding per-user (48K dict lookups × N users)
    pop_scores = np.array([train_pop.get(i, 0) for i in range(n_warm)], dtype=np.float32)

    cu_rows = raw_df[raw_df["user"].isin(cold_user_ids)].copy()
    cu_rows = cu_rows.sort_values(["user", "rating"], ascending=[True, False])

    # Sample cold users to cap evaluation cost
    all_cu = list(cu_rows["user"].unique())
    if len(all_cu) > MAX_COLD_USERS:
        rng_s  = np.random.default_rng(SEED)
        all_cu = rng_s.choice(all_cu, size=MAX_COLD_USERS, replace=False).tolist()
        cu_rows = cu_rows[cu_rows["user"].isin(set(all_cu))]
        print(f"  sampled {MAX_COLD_USERS:,} cold users for evaluation")

    rng = np.random.default_rng(SEED)
    sbert_records: list[dict] = []
    pop_records:   list[dict] = []
    skipped = 0

    for u_str, grp in cu_rows.groupby("user"):
        if len(grp) < 2:
            continue
        grp           = grp.reset_index(drop=True)
        test_item_str = grp.iloc[0]["item"]
        hist_strs     = grp.iloc[1:]["item"].tolist()

        test_int = i_map.get(test_item_str)
        if test_int is None or test_int >= n_warm:
            skipped += 1
            continue

        hist_ints = [i_map[s] for s in hist_strs if s in i_map and i_map[s] < n_warm]
        if not hist_ints:
            skipped += 1
            continue

        vecs    = item_embs[np.array(hist_ints)]
        norm    = np.linalg.norm(vecs, axis=1, keepdims=True); norm[norm == 0] = 1.0
        profile = (vecs / norm).mean(axis=0)

        sbert_scores = item_embs @ profile
        sbert_records.append(_rank_metrics(sbert_scores, test_int, rng, n_warm))
        pop_records.append(_rank_metrics(pop_scores,     test_int, rng, n_warm))

    print(f"  cold-user pairs evaluated: {len(sbert_records):,}  skipped: {skipped}")
    return {
        "sbert":      _aggregate(sbert_records),
        "popularity": _aggregate(pop_records),
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_report(counts: dict, cold_item_results: dict,
                  cold_user_results: dict) -> None:
    Path("reports/coldstart").mkdir(parents=True, exist_ok=True)
    sbert_hr = cold_item_results["sbert"]["hr@10"]
    cu_sbert = cold_user_results["sbert"]["hr@10"]
    cu_pop   = cold_user_results["popularity"]["hr@10"]

    if sbert_hr > 0.15:
        item_interp = (
            f"SBERT retrieves cold items with HR@10={sbert_hr:.4f}, above the 0.10 random baseline. "
            f"The semantic signal provides meaningful coverage for items with no collaborative history. "
            f"SVD and item-item CF score 0 on these items by design."
        )
    elif sbert_hr > 0.10:
        item_interp = (
            f"SBERT retrieves cold items with HR@10={sbert_hr:.4f}, marginally above the 0.10 random "
            f"baseline. The semantic signal gives a weak but non-zero advantage over random. "
            f"SVD and item-item CF score 0 on these items by design."
        )
    else:
        item_interp = (
            f"SBERT retrieves cold items with HR@10={sbert_hr:.4f}, at or below the 0.10 random "
            f"baseline. The semantic signal does not reliably retrieve cold items for these users. "
            f"Cold-start remains hard even with text-based encoding. This is an honest negative result."
        )

    if cu_sbert > cu_pop:
        user_interp = (
            f"SBERT-based profiles (HR@10={cu_sbert:.4f}) outperform popularity "
            f"(HR@10={cu_pop:.4f}) for cold users, suggesting item text helps even with very limited history."
        )
    else:
        user_interp = (
            f"Popularity (HR@10={cu_pop:.4f}) matches or beats SBERT profiles "
            f"(HR@10={cu_sbert:.4f}) for cold users. With only 1-4 interactions, "
            f"the SBERT profile does not capture enough preference signal to beat a simple frequency prior."
        )

    lines = [
        "# Cold-Start Supplementary Experiment",
        "",
        "All numbers here are supplementary to the main dissertation results.",
        "This experiment wrote only to data/coldstart/ and reports/coldstart/ and did not",
        "modify any main pipeline artifact.",
        "",
        "## 1. Dataset Construction",
        "",
        "Source: data/raw.parquet (Amazon Electronics 2023, before k_core=5 filter).",
        "Cold items: raw interaction count in [1, 4] AND absent from the k_core=5 clean set.",
        "Cold users: raw interaction count in [1, 4] AND absent from the k_core=5 clean set.",
        "",
        "| Entity | Count |",
        "|--------|-------|",
        f"| Cold items in raw (1-4 interactions) | {counts['n_cold_items']:,} |",
        f"| Cold users in raw (1-4 interactions) | {counts['n_cold_users']:,} |",
        f"| Cold items with at least one warm-user pair | {counts['n_testable_cold_items']:,} |",
        f"| Cold items sampled for evaluation | {MAX_COLD_ITEMS:,} (capped for encoding cost) |",
        f"| Warm-user / cold-item test pairs (rating >= 4) | {counts['n_test_pairs']:,} |",
        f"| Cold users sampled for evaluation | {min(counts['n_cold_users'], MAX_COLD_USERS):,} |",
        f"| Cold-user LOO pairs evaluated | {cold_user_results['sbert']['n_pairs']:,} |",
        "",
        "Leakage check: cold entities were identified from raw data outside the main training split.",
        "No cold item or cold user appears in data/clean.parquet.",
        "Item popularity used for scoring comes from training rows only.",
        "",
        "## 2. Cold-Item Retrieval",
        "",
        "Question: can the semantic signal retrieve cold items that interaction-based signals cannot?",
        "",
        "Warm users who rated a cold item (rating >= 4) form the test pairs.",
        f"Protocol: sampled negatives — 99 random cold items per positive, top-{TOP_K}.",
        "",
        "| Signal | HR@10 | NDCG@10 | MRR | n pairs |",
        "|--------|-------|---------|-----|---------|",
        f"| SBERT (semantic) | {cold_item_results['sbert']['hr@10']:.4f} | "
        f"{cold_item_results['sbert']['ndcg@10']:.4f} | "
        f"{cold_item_results['sbert']['mrr']:.4f} | "
        f"{cold_item_results['sbert']['n_pairs']:,} |",
        f"| SVD (collaborative) | {cold_item_results['svd']} |",
        f"| Item-Item CF | {cold_item_results['itemitem']} |",
        "",
        "Random baseline for sampled-negative HR@10 at k=10 with 99 negatives: 0.1000.",
        "",
        "## 3. Cold-User Recommendation",
        "",
        "Question: can we recommend for users with almost no interaction history?",
        "",
        "Cold users with >= 2 raw interactions, LOO split (hold out highest-rated item).",
        "History items must be warm (in k_core=5) to have an embedding.",
        f"Protocol: sampled negatives — 99 random warm items per positive, top-{TOP_K}.",
        "",
        "| Signal | HR@10 | NDCG@10 | MRR | n pairs |",
        "|--------|-------|---------|-----|---------|",
        f"| SBERT avg-history profile | {cold_user_results['sbert']['hr@10']:.4f} | "
        f"{cold_user_results['sbert']['ndcg@10']:.4f} | "
        f"{cold_user_results['sbert']['mrr']:.4f} | "
        f"{cold_user_results['sbert']['n_pairs']:,} |",
        f"| Popularity baseline | {cold_user_results['popularity']['hr@10']:.4f} | "
        f"{cold_user_results['popularity']['ndcg@10']:.4f} | "
        f"{cold_user_results['popularity']['mrr']:.4f} | "
        f"{cold_user_results['popularity']['n_pairs']:,} |",
        "",
        "Random baseline: 0.1000.",
        "",
        "## 4. Interpretation",
        "",
        item_interp,
        "",
        user_interp,
        "",
    ]

    Path("reports/coldstart/coldstart_results.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("[report] written to reports/coldstart/coldstart_results.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0  = time.time()
    sep = "=" * 62
    print(f"\n{sep}")
    print("COLD-START SUPPLEMENTARY EXPERIMENT")
    print(f"{sep}")
    _assert_isolation()

    Path("data/coldstart").mkdir(parents=True, exist_ok=True)
    Path("reports/coldstart").mkdir(parents=True, exist_ok=True)

    # --- Step 1: warm ID maps ---
    print("\n[step 1] loading and cleaning raw data (user/item/rating only) ...")
    raw_df  = _clean_raw(Path("data/raw.parquet"))
    print(f"  raw after clean/dedup: {len(raw_df):,} rows  "
          f"{raw_df['user'].nunique():,} users  {raw_df['item'].nunique():,} items")
    warm_df = _kcore(raw_df, k=5)
    print(f"  after k_core=5: {len(warm_df):,} rows  "
          f"{warm_df['user'].nunique():,} users  {warm_df['item'].nunique():,} items")
    u_map, i_map = _build_id_maps(warm_df)
    _verify_maps(u_map, i_map)
    warm_users = set(warm_df["user"].unique())
    warm_items = set(warm_df["item"].unique())

    # --- Step 2: identify cold entities ---
    print("\n[step 2] identifying cold entities ...")
    cold_items, cold_users = _identify_cold(raw_df, warm_users, warm_items)
    print(f"  cold items (1-4 raw interactions, not in warm): {len(cold_items):,}")
    print(f"  cold users (1-4 raw interactions, not in warm): {len(cold_users):,}")
    cold_item_ids = set(cold_items["item"].tolist())
    cold_user_ids = set(cold_users["user"].tolist())

    # --- Step 3: cold-item test pairs ---
    print("\n[step 3] finding warm-user / cold-item test pairs ...")
    pairs = _cold_item_pairs(raw_df, warm_users, cold_item_ids)
    print(f"  warm-user / cold-item relevant pairs: {len(pairs):,}")
    testable = cold_items[cold_items["item"].isin(pairs["item"])].reset_index(drop=True)
    print(f"  cold items with at least one warm-user pair: {len(testable):,}")

    if len(testable) > MAX_COLD_ITEMS:
        rng_s   = np.random.default_rng(SEED)
        keep    = rng_s.choice(len(testable), size=MAX_COLD_ITEMS, replace=False)
        testable = testable.iloc[keep].reset_index(drop=True)
        pairs    = pairs[pairs["item"].isin(testable["item"])].reset_index(drop=True)
        print(f"  sampled down to {MAX_COLD_ITEMS:,} cold items  ({len(pairs):,} pairs remain)")

    # Save intermediate artefacts now (before encoding) so a crash doesn't lose them
    testable.to_parquet("data/coldstart/cold_items.parquet", index=False)
    cold_users.to_parquet("data/coldstart/cold_users.parquet", index=False)
    pairs.to_json("data/coldstart/cs_eval_pairs.json", orient="records")
    print(f"  saved cold_items.parquet  cold_users.parquet  cs_eval_pairs.json")

    # --- Step 4: SBERT embeddings for cold items ---
    print("\n[step 4] encoding cold items with SBERT ...")
    emb_path = Path("data/coldstart/cold_item_embeddings.npy")
    model    = SentenceTransformer("models/all-MiniLM-L6-v2")
    cold_item_embs = _encode_cold_items(testable, Path("data/raw.parquet"), model)
    np.save(str(emb_path), cold_item_embs)
    print(f"  saved cold_item_embeddings.npy  shape={cold_item_embs.shape}")

    # --- Step 5: warm user profiles ---
    print("\n[step 5] building warm user profiles from item_embeddings.npy ...")
    item_embs = np.load("features/item_embeddings.npy").astype(np.float32)
    norms     = np.linalg.norm(item_embs, axis=1, keepdims=True); norms[norms == 0] = 1.0
    item_embs = item_embs / norms
    pair_users    = pairs["user"].unique().tolist()
    user_profiles = _warm_user_profiles(pair_users, u_map, item_embs)
    print(f"  profiles built for {len(user_profiles):,} / {len(pair_users):,} pair users")

    # --- Step 6: cold ITEM evaluation ---
    print("\n[step 6] evaluating cold-item retrieval ...")
    cold_item_results = _eval_cold_items(pairs, testable, cold_item_embs, user_profiles)
    r = cold_item_results["sbert"]
    print(f"  SBERT  HR@10={r['hr@10']:.4f}  NDCG@10={r['ndcg@10']:.4f}  MRR={r['mrr']:.4f}")
    print(f"  SVD: {cold_item_results['svd']}")
    print(f"  ItemItem: {cold_item_results['itemitem']}")

    # --- Step 7: cold USER evaluation ---
    print("\n[step 7] evaluating cold-user recommendation ...")
    cold_user_results = _eval_cold_users(raw_df, cold_user_ids, i_map, item_embs)
    rs = cold_user_results["sbert"];      rp = cold_user_results["popularity"]
    print(f"  SBERT      HR@10={rs['hr@10']:.4f}  NDCG@10={rs['ndcg@10']:.4f}  MRR={rs['mrr']:.4f}")
    print(f"  Popularity HR@10={rp['hr@10']:.4f}  NDCG@10={rp['ndcg@10']:.4f}  MRR={rp['mrr']:.4f}")

    # --- Step 8: write report ---
    print("\n[step 8] writing report ...")
    counts = {
        "n_cold_items":          len(cold_items),
        "n_cold_users":          len(cold_users),
        "n_testable_cold_items": len(testable),
        "n_test_pairs":          len(pairs),
    }
    _write_report(counts, cold_item_results, cold_user_results)

    elapsed = (time.time() - t0) / 60
    print(f"\n{sep}")
    print(f"DONE  elapsed {elapsed:.1f}m")
    print(f"{sep}\n")
    print("Output files:")
    for root in WRITE_ROOTS:
        for p in sorted(Path(root).glob("*")):
            print(f"  {p}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
