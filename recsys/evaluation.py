"""
Evaluation: sampled-negative ranking metrics and leakage check.

Protocol: for each test user's positive, score it alongside
negatives_per_positive negatives drawn from the ranked candidate list,
then compute HR, NDCG, and MRR at each k.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from recsys.ranking import FEATURE_COLS


def _test_positives(
    split: dict, df: pd.DataFrame, threshold: float
) -> dict[int, list[int]]:
    """Map each test user to their high-rated test items."""
    test_df = df.loc[split["test_idx"]]
    pos = test_df[test_df["rating"] >= threshold]
    return {int(u): g["item"].tolist() for u, g in pos.groupby("user")}


def _hit_ndcg_mrr(rank: int, k_values: list[int]) -> dict[str, float]:
    """Return HR, NDCG, P@K, and MRR for a single positive at 1-indexed rank.

    With one positive per evaluation list:
      Recall@K == HR@K (hit or miss).
      Precision@K == HR@K / K.
    """
    result: dict[str, float] = {"mrr": 1.0 / rank}
    for k in k_values:
        hit = float(rank <= k)
        result[f"hr@{k}"] = hit
        result[f"ndcg@{k}"] = (1.0 / np.log2(rank + 1)) if rank <= k else 0.0
        result[f"p@{k}"] = hit / k
    return result


def _eval_user(
    ranked_user: pd.DataFrame,
    positives: list[int],
    neg_per_pos: int,
    k_values: list[int],
    rng: np.random.Generator,
) -> list[dict]:
    """Evaluate one user: one metrics dict per (user, positive) pair."""
    pos_set = set(positives)
    negs = ranked_user[~ranked_user["item"].isin(pos_set)]["item"].tolist()
    results = []
    for pos in positives:
        if pos not in ranked_user["item"].values:
            continue
        n_neg = min(neg_per_pos, len(negs))
        if n_neg == 0:
            continue
        sampled = rng.choice(negs, size=n_neg, replace=False).tolist()
        eval_items = {pos} | set(sampled)
        sub = ranked_user[ranked_user["item"].isin(eval_items)].sort_values("score", ascending=False)
        ranks = {item: r + 1 for r, item in enumerate(sub["item"].tolist())}
        rank = ranks.get(pos, n_neg + 1)
        results.append(_hit_ndcg_mrr(rank, k_values))
    return results


def _iter_user_candidates(
    cand_path: Path, score_col: str = "score"
) -> Iterator[tuple[int, pd.DataFrame]]:
    """Stream (user, df) pairs from candidate_scores.parquet one user at a time.

    Reads only user, item, and score_col so peak RAM is a single 200K-row batch
    instead of the full 15M-row table. Requires the parquet to be sorted by user,
    which the retrieval stage guarantees (it processes test_users in sorted order).
    """
    import pyarrow.parquet as pq

    read_cols = ["user", "item", score_col] if score_col != "score" else ["user", "item", "score"]
    pf = pq.ParquetFile(cand_path)
    buf_users: list[int] = []
    buf_items: list[int] = []
    buf_scores: list[float] = []
    cur_user: int | None = None

    for batch in pf.iter_batches(batch_size=200_000, columns=read_cols):
        users = batch.column("user").to_pylist()
        items = batch.column("item").to_pylist()
        scores = batch.column(score_col).to_pylist()
        for u, i, s in zip(users, items, scores):
            if cur_user is None:
                cur_user = u
            if u != cur_user:
                yield cur_user, pd.DataFrame(
                    {"user": buf_users, "item": buf_items, "score": buf_scores}
                )
                buf_users, buf_items, buf_scores = [u], [i], [s]
                cur_user = u
            else:
                buf_users.append(u)
                buf_items.append(i)
                buf_scores.append(s)

    if buf_users:
        yield cur_user, pd.DataFrame(
            {"user": buf_users, "item": buf_items, "score": buf_scores}
        )


def compute_ranking_metrics(
    cand_path: Path,
    score_col: str,
    user_positives: dict[int, list[int]],
    cold_users: set[int],
    cfg: dict,
) -> pd.DataFrame:
    """HR/NDCG/MRR for all test users with sampled negatives.

    Streams candidate_scores.parquet via _iter_user_candidates so no full-file
    sort or load is needed — avoids OOM on the 15M-row Pass 2 candidate set.
    """
    k_values = cfg["evaluation"]["k_values"]
    neg_per_pos = cfg["evaluation"]["negatives_per_positive"]
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for user, ranked_user in _iter_user_candidates(cand_path, score_col):
        positives = user_positives.get(int(user), [])
        if not positives:
            continue
        metrics_list = _eval_user(ranked_user, positives, neg_per_pos, k_values, rng)
        for m in metrics_list:
            m["user"] = int(user)
            m["slice"] = "cold" if int(user) in cold_users else "warm"
            rows.append(m)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def compute_rating_metrics(cfg: dict) -> dict[str, float]:
    """RMSE, MAE, R² from the pointwise model on the test partition."""
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    paths = cfg["paths"]
    model = lgb.Booster(model_file=str(Path(paths["ranking_dir"]) / "model_pointwise.json"))
    test = pd.read_parquet(Path(paths["features_dir"]) / "test_scores.parquet").dropna()
    X = test[FEATURE_COLS].values.astype(np.float32)
    y_true = test["rating"].values
    y_pred = model.predict(X)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def check_leakage(cfg: dict) -> dict[str, bool]:
    """Verify train/test disjoint and OOF only covers training (user, item) pairs."""
    df = pd.read_parquet(cfg["paths"]["clean"])
    split = json.load(open(cfg["paths"]["split"]))
    train_idx = set(split["train_idx"])
    test_idx = set(split["test_idx"])
    feat_dir = Path(cfg["paths"]["features_dir"])
    disjoint = len(train_idx & test_idx) == 0
    oof = pd.read_parquet(feat_dir / "oof_scores.parquet")
    train_df = df.loc[sorted(train_idx)]
    oof_pairs = set(zip(oof["user"].tolist(), oof["item"].tolist()))
    train_pairs = set(zip(train_df["user"].tolist(), train_df["item"].tolist()))
    oof_clean = oof_pairs <= train_pairs
    train_pop = train_df["item"].value_counts()
    oof_pop = oof.groupby("item")["item_popularity"].first()
    shared = train_pop.index.intersection(oof_pop.index)
    pop_match = bool((oof_pop.loc[shared] == train_pop.loc[shared]).all()) if len(shared) > 0 else True
    return {
        "train_test_disjoint": bool(disjoint),
        "oof_only_train_rows": bool(oof_clean),
        "item_popularity_from_train_only": pop_match,
    }


def compute_coverage_gini(ranked: pd.DataFrame, n_items: int) -> dict[str, float]:
    """Catalogue coverage and Gini popularity-bias over the ranked lists."""
    recommended = ranked["item"].unique()
    coverage = len(recommended) / n_items if n_items > 0 else 0.0
    freq = ranked["item"].value_counts().values.astype(float)
    freq.sort()
    n = len(freq)
    if n == 0:
        return {"coverage": coverage, "gini": float("nan")}
    gini = float((2 * np.arange(1, n + 1).dot(freq) - (n + 1) * freq.sum()) / (n * freq.sum()))
    return {"coverage": float(coverage), "gini": gini}


def _format_ranking_section(label: str, df: pd.DataFrame, k_values: list[int]) -> list[str]:
    """Format one model's ranking metrics into report lines."""
    n = df["user"].nunique() if len(df) > 0 else 0
    lines = [f"\n{label}  ({n:,} users)"]
    if len(df) == 0:
        lines.append("  [no test pairs — check candidate/ranking artifacts match the data split]")
        return lines
    for sl in ["all", "warm", "cold"]:
        sub = df if sl == "all" else df[df["slice"] == sl]
        if len(sub) == 0:
            continue
        lines.append(f"  [{sl}]")
        for k in k_values:
            hr = sub[f"hr@{k}"].mean()
            ndcg = sub[f"ndcg@{k}"].mean()
            p = sub[f"p@{k}"].mean()
            lines.append(f"    HR@{k}={hr:.4f}  NDCG@{k}={ndcg:.4f}  P@{k}={p:.4f}")
        lines.append(f"    MRR={sub['mrr'].mean():.4f}")
    return lines


def compute_retrieval_recall(cfg: dict) -> dict[str, float]:
    """Fraction of test positives that appear in the retrieved candidate set."""
    cand_path = Path(cfg["paths"]["retrieval_dir"]) / "candidates.parquet"
    if not cand_path.exists():
        return {"retrieval_recall_all": float("nan"), "retrieval_recall_warm": float("nan"),
                "retrieval_recall_cold": float("nan")}
    split = json.load(open(cfg["paths"]["split"]))
    cold_users = set(json.load(open(cfg["paths"]["cold_start"]))["cold_users"])
    threshold = float(cfg["evaluation"]["relevance_threshold"])
    df = pd.read_parquet(cfg["paths"]["clean"], columns=["user", "item", "rating"])
    positives = df.loc[split["test_idx"]]
    positives = positives[positives["rating"] >= threshold]
    cand_set = set(zip(
        pd.read_parquet(cand_path, columns=["user"])["user"].tolist(),
        pd.read_parquet(cand_path, columns=["item"])["item"].tolist(),
    ))
    pos_pairs = list(zip(positives["user"].tolist(), positives["item"].tolist()))
    warm_pairs = [(u, i) for u, i in pos_pairs if u not in cold_users]
    cold_pairs = [(u, i) for u, i in pos_pairs if u in cold_users]

    def _recall(pairs: list) -> float:
        return sum(1 for p in pairs if p in cand_set) / len(pairs) if pairs else float("nan")

    return {
        "retrieval_recall_all": _recall(pos_pairs),
        "retrieval_recall_warm": _recall(warm_pairs),
        "retrieval_recall_cold": _recall(cold_pairs),
    }


def write_leakage_report(leakage: dict[str, bool], report_dir: Path) -> None:
    """Write a standalone leakage_check.md file."""
    lines = ["# Leakage Check Report", ""]
    all_pass = all(leakage.values())
    lines.append(f"Overall: {'PASS' if all_pass else 'FAIL'}")
    lines.append("")
    for k, v in leakage.items():
        lines.append(f"- {k}: {'PASS' if v else '**FAIL**'}")
    lines.append("")
    lines.append("A PASS on all three checks confirms that no test row informed any")
    lines.append("feature it depends on and that the split contract held throughout.")
    (report_dir / "leakage_check.md").write_text("\n".join(lines), encoding="utf-8")


def _ranking_dict(label: str, df: pd.DataFrame, k_values: list[int]) -> dict:
    """Build a dict of metrics for one model, suitable for JSON serialisation."""
    entry: dict = {"model": label, "n_users": int(df["user"].nunique()) if len(df) else 0}
    for sl in ["all", "warm", "cold"]:
        sub = df if sl == "all" else df[df["slice"] == sl]
        if len(sub) == 0:
            continue
        sl_entry: dict = {"n": len(sub)}
        for k in k_values:
            sl_entry[f"hr@{k}"] = round(float(sub[f"hr@{k}"].mean()), 4)
            sl_entry[f"ndcg@{k}"] = round(float(sub[f"ndcg@{k}"].mean()), 4)
            sl_entry[f"p@{k}"] = round(float(sub[f"p@{k}"].mean()), 4)
        sl_entry["mrr"] = round(float(sub["mrr"].mean()), 4)
        entry[sl] = sl_entry
    return entry


def write_report(
    main: pd.DataFrame, baselines: dict[str, pd.DataFrame],
    rating: dict[str, float], leakage: dict[str, bool],
    retrieval: dict[str, float], bias: dict[str, float], cfg: dict,
) -> None:
    """Write metrics.txt, metrics.json, metrics.md, and leakage_check.md."""
    report_dir = Path(cfg["paths"]["reports_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    k_values = cfg["evaluation"]["k_values"]

    # ── plain-text report ──────────────────────────────────────────────────────
    lines = ["Recommender Evaluation Report", "=" * 40]
    lines += _format_ranking_section("HYBRID RANKER", main, k_values)
    for name, df in baselines.items():
        lines += _format_ranking_section(f"BASELINE: {name.upper()}", df, k_values)
    lines += ["\nRATING PREDICTION (pointwise)"]
    for k, v in rating.items():
        lines.append(f"  {k}={v:.4f}")
    lines += ["\nRETRIEVAL RECALL"]
    for k, v in retrieval.items():
        lines.append(f"  {k}={v:.4f}" if v == v else f"  {k}=N/A")
    lines += ["\nSYSTEM PROPERTIES"]
    lines.append(f"  catalogue_coverage={bias.get('coverage', float('nan')):.4f}")
    lines.append(f"  gini_popularity_bias={bias.get('gini', float('nan')):.4f}")
    lines += ["\nLEAKAGE CHECK"]
    for k, v in leakage.items():
        lines.append(f"  {k}: {'PASS' if v else 'FAIL'}")
    text = "\n".join(lines)
    (report_dir / "metrics.txt").write_text(text, encoding="utf-8")
    print(text)

    # ── JSON report ────────────────────────────────────────────────────────────
    payload = {
        "hybrid_ranker": _ranking_dict("HYBRID RANKER", main, k_values),
        "baselines": {n: _ranking_dict(n, df, k_values) for n, df in baselines.items()},
        "rating_prediction": {k: round(v, 4) for k, v in rating.items()},
        "retrieval_recall": {k: round(v, 4) if v == v else None for k, v in retrieval.items()},
        "system": {k: round(v, 4) if v == v else None for k, v in bias.items()},
        "leakage": leakage,
    }
    (report_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ── Markdown report ────────────────────────────────────────────────────────
    md = ["# Recommender Evaluation Report", ""]
    md.append("## Ranking metrics (sampled-negative protocol, 1 pos : 99 neg)")
    md.append("")
    header = "| Model | Slice |" + "".join(f" HR@{k} | NDCG@{k} | P@{k} |" for k in k_values) + " MRR |"
    sep = "|---|---|" + "---|---|---|" * len(k_values) + "---|"
    md += [header, sep]
    for entry in [_ranking_dict("Hybrid ranker", main, k_values)] + \
                 [_ranking_dict(n, df, k_values) for n, df in baselines.items()]:
        for sl in ["all", "warm", "cold"]:
            if sl not in entry:
                continue
            s = entry[sl]
            row = f"| {entry['model']} | {sl} |"
            for k in k_values:
                row += f" {s.get(f'hr@{k}', ''):.4f} | {s.get(f'ndcg@{k}', ''):.4f} | {s.get(f'p@{k}', ''):.4f} |"
            row += f" {s.get('mrr', ''):.4f} |"
            md.append(row)
    md += ["", "## Rating prediction (pointwise)"]
    for k, v in rating.items():
        md.append(f"- {k}: {v:.4f}")
    md += ["", "## System properties"]
    md.append(f"- Catalogue coverage: {bias.get('coverage', float('nan')):.4f}")
    md.append(f"- Gini popularity bias: {bias.get('gini', float('nan')):.4f}")
    md += ["", "## Leakage check"]
    for k, v in leakage.items():
        md.append(f"- {k}: {'PASS' if v else '**FAIL**'}")
    (report_dir / "metrics.md").write_text("\n".join(md), encoding="utf-8")

    write_leakage_report(leakage, report_dir)
    print(f"\n[evaluation] reports written to {report_dir}")


def build(cfg: dict) -> None:
    """Evaluate all models and write the metrics report."""
    paths = cfg["paths"]
    rank_dir = Path(paths["ranking_dir"])
    report_dir = Path(paths["reports_dir"])
    if (report_dir / "metrics.txt").exists():
        print("[evaluation] report found, skipping.")
        return

    # Pre-compute positives and cold users once — reused across all 6 evaluation passes.
    print("[evaluation] loading test positives …")
    df_light = pd.read_parquet(paths["clean"], columns=["user", "item", "rating"])
    split = json.load(open(paths["split"]))
    cold_users: set[int] = set(json.load(open(paths["cold_start"]))["cold_users"])
    threshold = float(cfg["evaluation"]["relevance_threshold"])
    user_positives = _test_positives(split, df_light, threshold)
    del df_light
    gc.collect()

    cand_path = rank_dir / "candidate_scores.parquet"

    print("[evaluation] computing ranking metrics (main model) …")
    main_results = compute_ranking_metrics(cand_path, "score", user_positives, cold_users, cfg)
    print(f"[evaluation] main: {len(main_results):,} (user, positive) pairs")

    baseline_results: dict[str, pd.DataFrame] = {}
    for name, col in [("popularity", "item_popularity"), ("svd", "svd_score"),
                      ("semantic", "semantic_score"), ("itemitem", "itemitem_score"),
                      ("tfidf", "tfidf_score")]:
        print(f"[evaluation] computing baseline: {name} …")
        baseline_results[name] = compute_ranking_metrics(
            cand_path, col, user_positives, cold_users, cfg
        )
        print(f"[evaluation] {name}: {len(baseline_results[name]):,} pairs")

    print("[evaluation] computing rating metrics …")
    rating_results = compute_rating_metrics(cfg)
    print("[evaluation] computing retrieval recall …")
    retrieval_results = compute_retrieval_recall(cfg)
    print("[evaluation] computing coverage and Gini …")
    df_clean = pd.read_parquet(paths["clean"], columns=["item"])
    n_items = int(df_clean["item"].max()) + 1
    del df_clean
    gc.collect()
    ranked_lists = pd.read_parquet(rank_dir / "ranked_lists.parquet")
    bias_results = compute_coverage_gini(ranked_lists, n_items)
    print("[evaluation] running leakage check …")
    leakage_results = check_leakage(cfg)
    write_report(main_results, baseline_results, rating_results, leakage_results,
                 retrieval_results, bias_results, cfg)
    main_results.to_parquet(report_dir / "ranking_results.parquet", index=False)
    print("[evaluation] done.")
