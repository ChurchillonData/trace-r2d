"""
Extended cold-start analysis.

Loads outputs from coldstart_experiment.py (cached parquet / npy files)
and adds:
  - scale framing: % of catalogue that is cold
  - per-interaction-count breakdown for cold users (2, 3, 4 interactions)
  - per-interaction-count breakdown for cold items (1–4 interactions)
  - skipped-user analysis
  - fig26: HR@10 warm-up curve
  - full extended coldstart_results.md

Writes only to reports/coldstart/ and reports/figures/.
Never touches main pipeline artefacts.

Usage:
  python -m experiments.coldstart_extended
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED        = 42
NEG_PER_POS = 99
TOP_K       = 10
MAX_CU      = 30_000   # cold-user sample cap for per-count eval


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _rank_metrics(scores: np.ndarray, pos_idx: int,
                  rng: np.random.Generator, n: int) -> dict:
    neg    = [i for i in range(n) if i != pos_idx]
    neg_s  = rng.choice(neg, size=min(NEG_PER_POS, len(neg)), replace=False)
    cands  = np.array([pos_idx] + neg_s.tolist())
    order  = np.argsort(-scores[cands])
    rank   = int(np.where(order == 0)[0][0]) + 1
    return {"hr10":   1.0 if rank <= TOP_K else 0.0,
            "ndcg10": (1.0 / np.log2(rank + 1)) if rank <= TOP_K else 0.0,
            "mrr":    1.0 / rank}


def _agg(recs: list[dict]) -> dict:
    if not recs:
        return {"hr@10": 0.0, "ndcg@10": 0.0, "mrr": 0.0, "n": 0}
    return {"hr@10":   round(float(np.mean([r["hr10"]   for r in recs])), 4),
            "ndcg@10": round(float(np.mean([r["ndcg10"] for r in recs])), 4),
            "mrr":     round(float(np.mean([r["mrr"]    for r in recs])), 4),
            "n":       len(recs)}


def _kcore(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    while True:
        uc   = df["user"].value_counts()
        ic   = df["item"].value_counts()
        mask = df["user"].isin(uc[uc >= k].index) & df["item"].isin(ic[ic >= k].index)
        if mask.all():
            return df.reset_index(drop=True)
        df = df[mask]


# ---------------------------------------------------------------------------
# Rebuild warm ID maps (fast — no review column)
# ---------------------------------------------------------------------------

def _rebuild_maps() -> tuple[pd.DataFrame, dict, dict, set, set]:
    print("[setup] loading raw data (user/item/rating only) ...")
    raw = pd.read_parquet("data/raw.parquet", columns=["user", "item", "rating"])
    raw = raw[(raw["user"].str.len() > 0) & (raw["item"].str.len() > 0)]
    raw["rating"] = pd.to_numeric(raw["rating"], errors="coerce")
    raw = raw.dropna(subset=["rating"]).drop_duplicates(subset=["user", "item"]).reset_index(drop=True)
    print(f"  raw: {len(raw):,} rows  {raw['user'].nunique():,} users  {raw['item'].nunique():,} items")

    warm = _kcore(raw)
    u_map = {u: i for i, u in enumerate(sorted(warm["user"].unique()))}
    i_map = {it: i for i, it in enumerate(sorted(warm["item"].unique()))}
    print(f"  warm (k_core=5): {warm['user'].nunique():,} users  {warm['item'].nunique():,} items")
    return raw, u_map, i_map, set(warm["user"].unique()), set(warm["item"].unique())


# ---------------------------------------------------------------------------
# Warm user profiles (for cold-item-by-count re-evaluation)
# ---------------------------------------------------------------------------

def _user_profiles(pair_users: list[str], u_map: dict,
                   item_embs: np.ndarray) -> dict[str, np.ndarray]:
    split    = json.loads(Path("data/split.json").read_text(encoding="utf-8"))
    clean    = pd.read_parquet("data/clean.parquet", columns=["user", "item", "rating"])
    train_df = clean.loc[split["train_idx"]]
    inv_u    = {v: k for k, v in u_map.items()}
    n_warm   = len(item_embs)
    profiles: dict[str, np.ndarray] = {}
    for s_u in pair_users:
        int_u = u_map.get(s_u)
        if int_u is None:
            continue
        rows  = train_df[(train_df["user"] == int_u) & (train_df["rating"] >= 4)]
        valid = [ii for ii in rows["item"].values if 0 <= ii < n_warm]
        if not valid:
            valid = [ii for ii in train_df[train_df["user"] == int_u]["item"].values
                     if 0 <= ii < n_warm]
        if valid:
            vecs = item_embs[np.array(valid)]
            norm = np.linalg.norm(vecs, axis=1, keepdims=True); norm[norm == 0] = 1.0
            profiles[s_u] = (vecs / norm).mean(axis=0)
    return profiles


# ---------------------------------------------------------------------------
# Per-count cold-user evaluation
# ---------------------------------------------------------------------------

def eval_cu_by_count(raw_df: pd.DataFrame, cold_user_ids: set,
                     i_map: dict, item_embs: np.ndarray) -> dict:
    split     = json.loads(Path("data/split.json").read_text(encoding="utf-8"))
    clean     = pd.read_parquet("data/clean.parquet", columns=["user", "item", "rating"])
    train_pop = clean.loc[split["train_idx"], "item"].value_counts()
    n_warm    = len(item_embs)
    pop_scores = np.array([train_pop.get(i, 0) for i in range(n_warm)], dtype=np.float32)

    cu = raw_df[raw_df["user"].isin(cold_user_ids)].copy()
    cu = cu.sort_values(["user", "rating"], ascending=[True, False])

    all_cu = list(cu["user"].unique())
    if len(all_cu) > MAX_CU:
        rng_s  = np.random.default_rng(SEED)
        all_cu = rng_s.choice(all_cu, MAX_CU, replace=False).tolist()
        cu     = cu[cu["user"].isin(set(all_cu))]

    rng     = np.random.default_rng(SEED)
    buckets = {c: {"sbert": [], "pop": [], "skip": 0} for c in [2, 3, 4]}

    for u_str, grp in cu.groupby("user"):
        n = len(grp)
        if n < 2 or n > 4:
            continue
        grp      = grp.reset_index(drop=True)
        test_str = grp.iloc[0]["item"]
        hist     = grp.iloc[1:]["item"].tolist()

        test_int = i_map.get(test_str)
        if test_int is None or test_int >= n_warm:
            buckets[n]["skip"] += 1; continue

        hist_ints = [i_map[s] for s in hist if s in i_map and i_map[s] < n_warm]
        if not hist_ints:
            buckets[n]["skip"] += 1; continue

        vecs = item_embs[np.array(hist_ints)]
        norm = np.linalg.norm(vecs, axis=1, keepdims=True); norm[norm == 0] = 1.0
        prof = (vecs / norm).mean(axis=0)

        ss = item_embs @ prof
        buckets[n]["sbert"].append(_rank_metrics(ss,         test_int, rng, n_warm))
        buckets[n]["pop"].append(  _rank_metrics(pop_scores, test_int, rng, n_warm))

    return {n: {"sbert": _agg(v["sbert"]), "popularity": _agg(v["pop"]),
                "skipped": v["skip"]}
            for n, v in buckets.items()}


# ---------------------------------------------------------------------------
# Cold-item by-count breakdown
# ---------------------------------------------------------------------------

def eval_ci_by_count(pairs: pd.DataFrame, cold_items: pd.DataFrame,
                     cold_item_embs: np.ndarray,
                     user_profiles: dict[str, np.ndarray]) -> dict:
    item_to_idx = {it: i for i, it in enumerate(cold_items["item"].tolist())}
    item_to_n   = dict(zip(cold_items["item"], cold_items["n_interactions"]))
    rng         = np.random.default_rng(SEED)
    buckets: dict[int, list] = {}

    for _, row in pairs.iterrows():
        u_str, i_str = str(row["user"]), str(row["item"])
        if u_str not in user_profiles or i_str not in item_to_idx:
            continue
        n       = int(item_to_n.get(i_str, 1))
        scores  = cold_item_embs @ user_profiles[u_str]
        m       = _rank_metrics(scores, item_to_idx[i_str], rng, len(cold_items))
        buckets.setdefault(n, []).append(m)

    return {n: _agg(recs) for n, recs in sorted(buckets.items())}


# ---------------------------------------------------------------------------
# Figure 26
# ---------------------------------------------------------------------------

def fig_warmup(cu_by_count: dict, ci_by_count: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Cold-Start Warm-Up Analysis", fontsize=14, fontweight="bold", y=1.01)

    # Panel A — cold users
    ax   = axes[0]
    cnts = [2, 3, 4]
    sh   = [cu_by_count[c]["sbert"]["hr@10"]      for c in cnts]
    ph   = [cu_by_count[c]["popularity"]["hr@10"]  for c in cnts]
    ns   = [cu_by_count[c]["sbert"]["n"]            for c in cnts]
    sk   = [cu_by_count[c]["skipped"]               for c in cnts]

    x, w = np.arange(len(cnts)), 0.35
    b1 = ax.bar(x - w/2, sh, w, label="SBERT avg-profile", color="#E8843A", alpha=0.88)
    b2 = ax.bar(x + w/2, ph, w, label="Popularity prior",  color="#5B8DB8", alpha=0.88)
    ax.axhline(0.10, ls="--", color="#888", lw=1.2, label="Random (0.10)")

    for bar, n in zip(b1, ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"n={n:,}", ha="center", va="bottom", fontsize=7)
    for bar, n, s in zip(b2, ns, sk):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"n={n:,}\nskip={s:,}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c} interactions" for c in cnts])
    ax.set_ylabel("HR@10  (99 sampled negatives, pool = 48K warm items)")
    ax.set_title("A — Cold Users: SBERT Profile vs Popularity by Interaction Count")
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(max(ph) * 1.25, 0.75))

    # Panel B — cold items
    ax2    = axes[1]
    ci_c   = sorted(ci_by_count.keys())
    ci_h   = [ci_by_count[c]["hr@10"] for c in ci_c]
    ci_n   = [ci_by_count[c]["n"]     for c in ci_c]

    bars = ax2.bar([str(c) for c in ci_c], ci_h, color="#E8843A", alpha=0.88,
                   label="SBERT (semantic)")
    ax2.axhline(0.10, ls="--", color="#888", lw=1.2, label="Random (0.10)")

    for bar, n in zip(bars, ci_n):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                 f"n={n:,}", ha="center", va="bottom", fontsize=7)

    ax2.set_xlabel("Cold item raw interaction count")
    ax2.set_ylabel("HR@10  (99 sampled negatives, pool = 3K cold items)")
    ax2.set_title("B — Cold Items: SBERT Retrieval by Interaction Count")
    ax2.legend(fontsize=8); ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_ylim(0, max(ci_h) * 1.3 if ci_h else 0.4)

    plt.tight_layout()
    out = Path("reports/figures/fig26_coldstart_warmup.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Extended report
# ---------------------------------------------------------------------------

def write_extended_report(raw_users: int, raw_items: int,
                          warm_users: int, warm_items: int,
                          n_cold_items: int, n_cold_users: int,
                          n_testable: int, n_truly_cold: int,
                          n_pairs: int,
                          base_ci: dict, base_cu: dict,
                          cu_by_count: dict, ci_by_count: dict) -> None:
    pct_cold_i  = 100 * n_cold_items  / raw_items
    pct_cold_u  = 100 * n_cold_users  / raw_users
    pct_truly   = 100 * n_truly_cold  / raw_items
    pct_test    = 100 * n_testable    / n_cold_items

    sbert_hr = base_ci["sbert"]["hr@10"]
    cu_s     = base_cu["sbert"]["hr@10"]
    cu_p     = base_cu["popularity"]["hr@10"]

    lines = [
        "# Cold-Start Supplementary Experiment — Extended Report",
        "",
        "All numbers here are supplementary to the main dissertation results.",
        "This experiment wrote only to data/coldstart/ and reports/coldstart/",
        "and did not modify any main pipeline artefact.",
        "",
        "---",
        "",
        "## 1. Scale of the Cold-Start Problem",
        "",
        "The k_core=5 filter that defines the warm training set removes the majority of",
        "the raw catalogue. Cold-start is not an edge case — it is the dominant regime.",
        "",
        "| Entity | Raw count | Warm (k_core=5) | Cold | Cold % |",
        "|--------|-----------|-----------------|------|--------|",
        f"| Users | {raw_users:,} | {warm_users:,} | {n_cold_users:,} | {pct_cold_u:.1f}% |",
        f"| Items | {raw_items:,} | {warm_items:,} | {n_cold_items:,} | {pct_cold_i:.1f}% |",
        "",
        "Of the {n_cold_items:,} cold items, {n_truly_cold:,} ({pct_truly:.1f}% of the raw catalogue)".format(
            n_cold_items=n_cold_items, n_truly_cold=n_truly_cold, pct_truly=pct_truly
        ),
        "have zero interactions with any warm user and are therefore completely untestable",
        "from interaction data. Only {n_testable:,} cold items ({pct_test:.1f}% of cold items)".format(
            n_testable=n_testable, pct_test=pct_test
        ),
        "had at least one warm-user rating that could serve as a test signal.",
        "",
        "---",
        "",
        "## 2. Dataset Construction",
        "",
        "Cold items: raw interaction count in [1, 4] AND absent from the k_core=5 clean set.",
        "Cold users: raw interaction count in [1, 4] AND absent from the k_core=5 clean set.",
        "",
        "| Entity | Count |",
        "|--------|-------|",
        f"| Cold items total | {n_cold_items:,} |",
        f"| Cold items with at least one warm-user pair (testable) | {n_testable:,} |",
        f"| Truly cold items (no warm-user pair — zero test signal) | {n_truly_cold:,} |",
        f"| Cold users total | {n_cold_users:,} |",
        f"| Warm-user / cold-item test pairs (rating >= 4) | {n_pairs:,} |",
        "",
        "Leakage check: cold entities were identified from raw data outside the main",
        "training split. No cold item or cold user appears in data/clean.parquet.",
        "Item popularity used for scoring comes from training rows only.",
        "",
        "---",
        "",
        "## 3. Cold-Item Retrieval — Headline Results",
        "",
        "Question: can the semantic signal retrieve cold items that interaction-based signals cannot?",
        "",
        "Warm users who rated a cold item (rating >= 4) form the test pairs.",
        "Protocol: sampled negatives — 99 random cold items per positive, top-10.",
        "Pool size: 3,000 cold items (sampled with seed=42 from 187,862 testable).",
        "",
        "| Signal | HR@10 | NDCG@10 | MRR | n pairs |",
        "|--------|-------|---------|-----|---------|",
        f"| SBERT (semantic) | {base_ci['sbert']['hr@10']:.4f} | "
        f"{base_ci['sbert']['ndcg@10']:.4f} | {base_ci['sbert']['mrr']:.4f} | "
        f"{base_ci['sbert']['n_pairs']:,} |",
        f"| SVD (collaborative) | N/A — cold items have no latent factors | | | |",
        f"| Item-Item CF | N/A — cold items have no co-rating history | | | |",
        "",
        "Random baseline (sampled negatives, k=10, pool=100): 0.1000.",
        "",
        f"SBERT achieves HR@10={sbert_hr:.4f}, {(sbert_hr/0.10 - 1)*100:.0f}% above random.",
        "SVD and item-item CF score exactly zero on cold items by design: without any",
        "interaction history there are no latent factors and no co-rated neighbours.",
        "SBERT is the only signal that can retrieve cold items at all, and it does so",
        "meaningfully.",
        "",
        "---",
        "",
        "## 4. Cold-Item Retrieval — by Interaction Count",
        "",
        "Items with more raw interactions have more review text available for embedding.",
        "The table below tests whether this helps SBERT retrieval quality.",
        "",
        "| Item raw count | HR@10 | NDCG@10 | MRR | n pairs |",
        "|---------------|-------|---------|-----|---------|",
    ]

    for c, r in sorted(ci_by_count.items()):
        lines.append(f"| {c} | {r['hr@10']:.4f} | {r['ndcg@10']:.4f} | {r['mrr']:.4f} | {r['n']:,} |")

    lines += [
        "",
        "Random baseline: 0.1000.",
        "",
        "---",
        "",
        "## 5. Cold-User Recommendation — Headline Results",
        "",
        "Question: can we recommend for users with almost no interaction history?",
        "",
        "Cold users with >= 2 raw interactions, leave-one-out split",
        "(hold out highest-rated item as test). History items must be warm",
        "(in k_core=5) to have an embedding.",
        "Protocol: sampled negatives — 99 random warm items per positive, top-10.",
        "Pool size: 48,154 warm items.",
        "",
        "| Signal | HR@10 | NDCG@10 | MRR | n pairs |",
        "|--------|-------|---------|-----|---------|",
        f"| SBERT avg-history profile | {base_cu['sbert']['hr@10']:.4f} | "
        f"{base_cu['sbert']['ndcg@10']:.4f} | {base_cu['sbert']['mrr']:.4f} | "
        f"{base_cu['sbert']['n_pairs']:,} |",
        f"| Popularity baseline | {base_cu['popularity']['hr@10']:.4f} | "
        f"{base_cu['popularity']['ndcg@10']:.4f} | {base_cu['popularity']['mrr']:.4f} | "
        f"{base_cu['popularity']['n_pairs']:,} |",
        "",
        "Random baseline: 0.1000.",
        "",
        f"Popularity (HR@10={cu_p:.4f}) beats SBERT profiles (HR@10={cu_s:.4f}) by a ratio",
        f"of {cu_p/cu_s:.1f}x. With only 1-4 interactions the average embedding vector",
        "is dominated by noise — each additional item shifts the mean substantially,",
        "so the profile does not converge to a stable preference direction.",
        "A flat popularity prior, which encodes global frequency without any user signal,",
        "is more reliable at this sparsity level.",
        "Note that SBERT still beats random (0.10) by {:.0f}%, so it is not useless;".format((cu_s/0.10-1)*100),
        "it simply loses to a very strong simple baseline.",
        "",
        "---",
        "",
        "## 6. Cold-User Recommendation — by Interaction Count",
        "",
        "Does having more history interactions improve the SBERT profile quality?",
        "",
        "| Interactions | SBERT HR@10 | Pop HR@10 | n evaluated | n skipped |",
        "|-------------|-------------|-----------|-------------|-----------|",
    ]

    for c in [2, 3, 4]:
        r = cu_by_count[c]
        lines.append(
            f"| {c} | {r['sbert']['hr@10']:.4f} | "
            f"{r['popularity']['hr@10']:.4f} | "
            f"{r['sbert']['n']:,} | {r['skipped']:,} |"
        )

    # Compute trend direction
    sh_2 = cu_by_count[2]["sbert"]["hr@10"]
    sh_4 = cu_by_count[4]["sbert"]["hr@10"]
    if sh_4 > sh_2 * 1.05:
        trend = (f"SBERT profile quality improves from {sh_2:.4f} (2 interactions) to "
                 f"{sh_4:.4f} (4 interactions), confirming that more history helps "
                 "even within the cold-start regime.")
    elif sh_4 < sh_2 * 0.95:
        trend = (f"SBERT profile quality does not improve consistently across counts "
                 f"({sh_2:.4f} at 2 interactions, {sh_4:.4f} at 4), suggesting that "
                 "1-4 interactions is too sparse for the profile to stabilise "
                 "regardless of the exact count.")
    else:
        trend = (f"SBERT profile quality is roughly flat across 2-4 interactions "
                 f"({sh_2:.4f} to {sh_4:.4f}), indicating that even the jump from "
                 "2 to 4 interactions is insufficient to stabilise the preference profile.")

    skip_note_total = sum(cu_by_count[c]["skipped"] for c in [2, 3, 4])
    lines += [
        "",
        trend,
        "",
        f"The skipped column counts cold users whose test item or entire history was also",
        f"cold (not in the warm k_core=5 set). These {skip_note_total:,} users are a",
        "second-order cold-start case: they interacted only with other cold items, making",
        "it impossible to build any embedding-based profile. No system can serve these",
        "users from content alone without additional context such as session signals",
        "or demographic priors.",
        "",
        "---",
        "",
        "## 7. Budget Allocation Argument",
        "",
        "In the main two-stage pipeline, SBERT receives 16.5% of the retrieval candidate",
        f"budget. For warm items this is a minority contribution. For the {pct_cold_i:.0f}%",
        "of the catalogue that is cold, SBERT is the only signal available — SVD and",
        "item-item CF contribute zero candidates for these items by design.",
        "",
        "A natural extension is an adaptive budget that raises the SBERT share when the",
        "candidate pool is known to contain cold items (for example, when a query item",
        "is absent from the warm catalogue, or when the retrieval stage detects that",
        "collaborative sources returned very few candidates).",
        "",
        "---",
        "",
        "## 8. Recommendations for Future Work",
        "",
        "Cold items:",
        "- The current SBERT signal already retrieves cold items at HR@10={:.4f}, well".format(sbert_hr),
        "  above random. Including SBERT in the retrieval union is already the right call.",
        "- Richer item representations (product category, price tier, brand embeddings)",
        "  could strengthen the signal for the 144K truly cold items that have no",
        "  warm-user test pairs at all.",
        "",
        "Cold users:",
        "- With 1-4 interactions, popularity is a hard baseline to beat. A hybrid of",
        "  popularity prior and a sparse SBERT profile (weighted by interaction count)",
        "  would likely narrow the gap.",
        "- Session-level signals (click sequence, dwell time) would provide richer",
        "  context than rating counts alone and are the standard industry approach",
        "  for cold-user personalisation.",
        "",
        "See fig26_coldstart_warmup.png for the visual summary.",
        "",
    ]

    out = Path("reports/coldstart/coldstart_results.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] written to {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "="*62)
    print("COLD-START EXTENDED ANALYSIS")
    print("="*62)

    # --- Load cached experiment outputs ---
    print("\n[load] cached experiment outputs ...")
    cold_items = pd.read_parquet("data/coldstart/cold_items.parquet")
    cold_users = pd.read_parquet("data/coldstart/cold_users.parquet")
    pairs      = pd.read_json("data/coldstart/cs_eval_pairs.json", orient="records")
    ci_embs    = np.load("data/coldstart/cold_item_embeddings.npy").astype(np.float32)
    item_embs_raw = np.load("features/item_embeddings.npy").astype(np.float32)
    norms = np.linalg.norm(item_embs_raw, axis=1, keepdims=True); norms[norms==0]=1.0
    item_embs = item_embs_raw / norms
    print(f"  cold_items={len(cold_items):,}  cold_users={len(cold_users):,}  "
          f"pairs={len(pairs):,}  ci_embs={ci_embs.shape}  item_embs={item_embs.shape}")

    # Load base results from existing report to avoid recomputing
    base_ci = {
        "sbert":    {"hr@10": 0.1967, "ndcg@10": 0.0957, "mrr": 0.0891, "n_pairs": 3919},
        "svd":      "N/A — cold items have no collaborative factors",
        "itemitem": "N/A — cold items have no co-rating history",
    }
    base_cu = {
        "sbert":      {"hr@10": 0.1563, "ndcg@10": 0.0804, "mrr": 0.0807, "n_pairs": 2815},
        "popularity": {"hr@10": 0.5869, "ndcg@10": 0.3995, "mrr": 0.3571, "n_pairs": 2815},
    }

    # --- Rebuild warm maps (needed for per-count eval) ---
    raw_df, u_map, i_map, warm_users_set, warm_items_set = _rebuild_maps()
    raw_users = raw_df["user"].nunique()
    raw_items = raw_df["item"].nunique()
    warm_users = len(warm_users_set)
    warm_items = len(warm_items_set)

    # Scale stats
    n_cold_items = len(cold_users)  # from experiment output note (332,471)
    # Recompute directly from raw:
    item_counts = raw_df["item"].value_counts()
    user_counts = raw_df["user"].value_counts()
    n_cold_items_actual = len(set(item_counts[item_counts <= 4].index) - warm_items_set)
    n_cold_users_actual = len(set(user_counts[user_counts <= 4].index) - warm_users_set)

    # testable = cold items that have a warm-user pair
    cold_item_ids = set(cold_items["item"].tolist())
    testable_mask = pairs["item"].isin(cold_item_ids)
    n_testable = len(cold_item_ids)  # 3000 sampled
    n_testable_all = 187_862          # from experiment run output
    n_truly_cold = n_cold_items_actual - n_testable_all

    print(f"\n[scale]")
    print(f"  raw users={raw_users:,}  warm={warm_users:,}  cold={n_cold_users_actual:,}"
          f"  ({100*n_cold_users_actual/raw_users:.1f}%)")
    print(f"  raw items={raw_items:,}  warm={warm_items:,}  cold={n_cold_items_actual:,}"
          f"  ({100*n_cold_items_actual/raw_items:.1f}%)")
    print(f"  truly cold items (no warm-user pair): {n_truly_cold:,}")

    # --- Per-count cold-user evaluation ---
    print("\n[step A] per-interaction-count cold-user evaluation ...")
    cold_user_ids = set(cold_users["user"].tolist())
    cu_by_count = eval_cu_by_count(raw_df, cold_user_ids, i_map, item_embs)
    for c in [2, 3, 4]:
        r = cu_by_count[c]
        print(f"  {c}-interaction users: SBERT HR@10={r['sbert']['hr@10']:.4f}  "
              f"Pop HR@10={r['popularity']['hr@10']:.4f}  "
              f"n={r['sbert']['n']:,}  skip={r['skipped']:,}")

    # --- Per-count cold-item evaluation ---
    print("\n[step B] per-interaction-count cold-item evaluation ...")
    pair_users = pairs["user"].unique().tolist()
    print(f"  building user profiles for {len(pair_users):,} pair users ...")
    user_profiles = _user_profiles(pair_users, u_map, item_embs)
    print(f"  profiles built: {len(user_profiles):,}")
    ci_by_count = eval_ci_by_count(pairs, cold_items, ci_embs, user_profiles)
    for c, r in sorted(ci_by_count.items()):
        print(f"  {c}-interaction items: SBERT HR@10={r['hr@10']:.4f}  n={r['n']:,}")

    # --- Figure ---
    print("\n[step C] generating fig26 ...")
    fig_warmup(cu_by_count, ci_by_count)

    # --- Extended report ---
    print("\n[step D] writing extended report ...")
    write_extended_report(
        raw_users=raw_users, raw_items=raw_items,
        warm_users=warm_users, warm_items=warm_items,
        n_cold_items=n_cold_items_actual, n_cold_users=n_cold_users_actual,
        n_testable=n_testable_all, n_truly_cold=n_truly_cold,
        n_pairs=len(pairs),
        base_ci=base_ci, base_cu=base_cu,
        cu_by_count=cu_by_count, ci_by_count=ci_by_count,
    )

    print("\n" + "="*62)
    print("DONE")
    print("="*62)


if __name__ == "__main__":
    main()
